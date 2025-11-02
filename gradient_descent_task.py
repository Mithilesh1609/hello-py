import asyncio
import json
import numpy as np
import logging
import time
from datetime import datetime
from collections.abc import Callable
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, TypedDict

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolUnionParam

MAX_TOKENS = 2000

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'gradient_descent_task_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PythonExpressionToolResult(TypedDict):
    result: Any
    error: str | None


class SubmitAnswerToolResult(TypedDict):
    answer: Any
    submitted: bool


def python_expression_tool(expression: str) -> PythonExpressionToolResult:
    """
    Tool that evaluates Python expressions using exec.
    Use print(...) to emit output; stdout will be captured and returned.
    Available imports: numpy as np, matplotlib.pyplot as plt
    """
    try:
        # Pre-import common ML libraries
        namespace = {
            'np': np,
            '__builtins__': __builtins__,
        }
        stdout = StringIO()
        with redirect_stdout(stdout):
            exec(expression, namespace, namespace)
        return {"result": stdout.getvalue(), "error": None}
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return {"result": None, "error": str(e)}


def submit_answer_tool(answer: Any) -> SubmitAnswerToolResult:
    """
    Tool for submitting the final answer.
    Expected answer format: {"final_x": float, "final_y": float, "converged": bool}
    """
    return {"answer": answer, "submitted": True}


def verify_gradient_descent_solution(result: dict) -> tuple[bool, str]:
    """
    Verify if the Positional Multi-Head Attention implementation is correct.
    
    Args:
        result: Dictionary with required keys for the attention task
    
    Returns:
        (success, message) tuple
    """
    try:
        if not isinstance(result, dict):
            return False, f"Expected dict, got {type(result)}"
        
        required_keys = {'position_encoded_queries', 'head1_attention_weights', 'head2_attention_weights',
                        'head1_output', 'head2_output', 'final_output', 'causal_mask_applied'}
        if not all(key in result for key in required_keys):
            missing = required_keys - set(result.keys())
            return False, f"Missing required keys: {missing}"
        
        # Reference implementation
        def reference_implementation():
            import numpy as np
            
            Q = np.array([[1.0, 0.5, -0.2], [0.3, -0.1, 0.8], [0.0, 0.4, -0.5], [-0.2, 0.2, 0.1]])
            K = np.array([[0.8, 0.2, -0.1], [0.1, -0.3, 0.6], [0.4, 0.7, 0.0], [0.0, 0.0, 0.9]])
            V = np.array([[1.5, -0.8, 0.3], [0.2, 1.0, -0.4], [-0.1, 0.6, 1.2], [0.9, 0.0, -0.7]])
            
            pos_emb = np.array([[0.1, 0.0, -0.1], [0.0, 0.1, 0.0], [-0.1, 0.0, 0.1], [0.0, -0.1, 0.0]])
            eps = 1e-8
            
            # 1. Position encoding
            Q_pos = Q + pos_emb
            K_pos = K + pos_emb
            V_pos = V + pos_emb
            
            # 2. Multi-head splitting
            # Head 1: Q,K dim=1, V dim=2
            Q1 = Q_pos[:, 0:1]  # Shape: (4, 1)
            K1 = K_pos[:, 0:1]  # Shape: (4, 1)
            V1 = V_pos[:, 0:2]  # Shape: (4, 2)
            
            # Head 2: Q,K dim=2, V dim=1
            Q2 = Q_pos[:, 1:3]  # Shape: (4, 2)
            K2 = K_pos[:, 1:3]  # Shape: (4, 2)
            V2 = V_pos[:, 2:3]  # Shape: (4, 1)
            
            def compute_attention(Q_h, K_h, V_h, apply_causal_mask=True):
                d_k = Q_h.shape[1]
                scale = 1.0 / np.sqrt(d_k)
                
                # Attention scores
                A = np.zeros((4, 4))
                for i in range(4):
                    for j in range(4):
                        A[i, j] = np.dot(Q_h[i], K_h[j]) * scale
                
                # Apply causal mask
                if apply_causal_mask:
                    for i in range(4):
                        for j in range(4):
                            if i < j:  # i > j in the problem means future positions, so i < j should be masked
                                A[i, j] = -np.inf
                
                # Softmax
                W = np.zeros((4, 4))
                for i in range(4):
                    exp_scores = np.exp(A[i] - np.max(A[i]))
                    W[i] = exp_scores / (np.sum(exp_scores) + eps)
                
                # Output
                O = np.zeros((4, V_h.shape[1]))
                for i in range(4):
                    O[i] = np.sum(W[i][:, np.newaxis] * V_h, axis=0)
                
                return A, W, O
            
            # 3. Per-head attention
            A1, W1, O1 = compute_attention(Q1, K1, V1)
            A2, W2, O2 = compute_attention(Q2, K2, V2)
            
            # 4. Concatenation
            # O1 is (4, 2), O2 is (4, 1), need to concatenate to (4, 3)
            O_final = np.concatenate([O1, O2], axis=1)
            
            return Q_pos, W1, W2, O1, O2, O_final
        
        # Get reference values
        ref_q_pos, ref_w1, ref_w2, ref_o1, ref_o2, ref_final = reference_implementation()
        
        # Extract user results
        try:
            user_q_pos = np.array(result['position_encoded_queries'])
            user_w1 = np.array(result['head1_attention_weights'])
            user_w2 = np.array(result['head2_attention_weights'])
            user_o1 = np.array(result['head1_output'])
            user_o2 = np.array(result['head2_output'])
            user_final = np.array(result['final_output'])
            user_mask = bool(result['causal_mask_applied'])
        except (ValueError, TypeError):
            return False, "Error converting results to proper types"
        
        # Check dimensions
        if user_q_pos.shape != (4, 3):
            return False, f"Expected position_encoded_queries shape (4, 3), got {user_q_pos.shape}"
        if user_w1.shape != (4, 4):
            return False, f"Expected head1_attention_weights shape (4, 4), got {user_w1.shape}"
        if user_w2.shape != (4, 4):
            return False, f"Expected head2_attention_weights shape (4, 4), got {user_w2.shape}"
        if user_o1.shape != (4, 2):
            return False, f"Expected head1_output shape (4, 2), got {user_o1.shape}"
        if user_o2.shape != (4, 1):
            return False, f"Expected head2_output shape (4, 1), got {user_o2.shape}"
        if user_final.shape != (4, 3):
            return False, f"Expected final_output shape (4, 3), got {user_final.shape}"
        
        # Tolerances (balanced - not too strict, not too lenient)
        tol_basic = 0.1
        tol_attention = 0.15
        tol_output = 0.2
        
        # Check each component
        q_pos_error = np.max(np.abs(user_q_pos - ref_q_pos))
        q_pos_correct = q_pos_error < tol_basic
        
        w1_error = np.max(np.abs(user_w1 - ref_w1))
        w1_correct = w1_error < tol_attention
        
        w2_error = np.max(np.abs(user_w2 - ref_w2))
        w2_correct = w2_error < tol_attention
        
        o1_error = np.max(np.abs(user_o1 - ref_o1))
        o1_correct = o1_error < tol_output
        
        o2_error = np.max(np.abs(user_o2 - ref_o2))
        o2_correct = o2_error < tol_output
        
        final_error = np.max(np.abs(user_final - ref_final))
        final_correct = final_error < tol_output
        
        # Success conditions (require good understanding of core concepts)
        basic_correct = q_pos_correct  # Position encoding
        attention_correct = w1_correct and w2_correct and user_mask  # Multi-head attention with masking
        output_correct = (o1_correct or o2_correct) and final_correct  # At least one head working + concatenation
        
        if basic_correct and attention_correct and output_correct:
            return True, f"SUCCESS: Complete multi-head attention. Errors: pos={q_pos_error:.3f}, w1={w1_error:.3f}, w2={w2_error:.3f}, final={final_error:.3f}"
        
        elif basic_correct and (w1_correct or w2_correct) and user_mask:
            return True, f"SUCCESS: Core concepts correct, one head working. Errors: pos={q_pos_error:.3f}, better_head={min(w1_error, w2_error):.3f}"
        
        elif basic_correct and not user_mask:
            return False, f"PARTIAL: Good position encoding but missing causal mask. Pos_err: {q_pos_error:.3f}"
        
        else:
            error_parts = []
            if not q_pos_correct:
                error_parts.append(f"position_encoding (err: {q_pos_error:.3f})")
            if not w1_correct:
                error_parts.append(f"head1_weights (err: {w1_error:.3f})")
            if not w2_correct:
                error_parts.append(f"head2_weights (err: {w2_error:.3f})")
            if not final_correct:
                error_parts.append(f"final_output (err: {final_error:.3f})")
            if not user_mask:
                error_parts.append("causal_mask_missing")
            
            return False, f"FAILURE: Incorrect {', '.join(error_parts)}"
            
    except (ValueError, TypeError, KeyError) as e:
        return False, f"Error parsing result: {e}"


async def run_agent_loop(
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    max_steps: int = 15,
    model: str = "claude-sonnet-4-5-20250929",
    verbose: bool = True,
) -> Any | None:
    """
    Runs an agent loop with the given prompt and tools.
    """
    client = AsyncAnthropic()
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]

    for step in range(max_steps):
        if verbose:
            print(f"\n=== Step {step + 1}/{max_steps} ===")

        response = await client.messages.create(
            model=model, max_tokens=MAX_TOKENS, tools=tools, messages=messages
        )

        assert response.stop_reason in ["max_tokens", "tool_use", "end_turn"], (
            f"unsupported stop_reason {response.stop_reason}"
        )
        if response.stop_reason == "max_tokens":
            print(f"Model reached max_tokens limit {MAX_TOKENS}.")

        has_tool_use = False
        tool_results = []
        submitted_answer = None

        for content in response.content:
            if content.type == "text":
                if verbose:
                    print(f"Assistant: {content.text}")
            elif content.type == "tool_use":
                has_tool_use = True
                tool_name = content.name

                if tool_name in tool_handlers:
                    if verbose:
                        print(f"Using tool: {tool_name}")

                    handler = tool_handlers[tool_name]
                    tool_input = content.input

                    if tool_name == "python_expression":
                        assert isinstance(tool_input, dict) and "expression" in tool_input
                        if verbose:
                            print("\nInput:")
                            print("```python")
                            print(tool_input["expression"])
                            print("```")
                        result = handler(tool_input["expression"])
                        if verbose:
                            print("\nOutput:")
                            print("```")
                            print(result["result"] if result["error"] is None else f"Error: {result['error']}")
                            print("```")
                    elif tool_name == "submit_answer":
                        assert isinstance(tool_input, dict) and "answer" in tool_input
                        result = handler(tool_input["answer"])
                        submitted_answer = result["answer"]
                    else:
                        result = (
                            handler(**tool_input)
                            if isinstance(tool_input, dict)
                            else handler(tool_input)
                        )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": json.dumps(result),
                        }
                    )

        if has_tool_use:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            if submitted_answer is not None:
                if verbose:
                    print(f"\nAgent submitted answer: {submitted_answer}")
                return submitted_answer
        else:
            if verbose:
                print("\nNo tool use in response, ending loop.")
            break

    if verbose:
        print(f"\nReached maximum steps ({max_steps}) without submitting answer.")
    return None


async def run_single_test(
    run_id: int,
    num_runs: int,
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    verbose: bool = False,
) -> tuple[int, bool, Any, str]:
    start_time = time.time()
    logger.info(f"Starting run {run_id}/{num_runs}")
    
    if verbose:
        print(f"\n\n{'=' * 20} RUN {run_id}/{num_runs} {'=' * 20}")

    try:
        result = await run_agent_loop(
            prompt=prompt,
            tools=tools,
            tool_handlers=tool_handlers,
            max_steps=15,
            verbose=verbose,
        )
        
        execution_time = time.time() - start_time
        logger.info(f"Run {run_id} completed in {execution_time:.2f}s")

        if result is None:
            logger.warning(f"Run {run_id} - No answer submitted")
            return run_id, False, None, "No answer submitted"

        success, message = verify_gradient_descent_solution(result)
        
        # Log the result details
        logger.info(f"Run {run_id} - Success: {success}")
        logger.info(f"Run {run_id} - Message: {message}")
        
        # Log submitted answer structure (without full content to avoid log spam)
        if isinstance(result, dict):
            keys_present = list(result.keys())
            logger.info(f"Run {run_id} - Submitted keys: {keys_present}")
            
            # Log key metrics if available
            try:
                if 'causal_mask_applied' in result:
                    logger.info(f"Run {run_id} - Causal mask applied: {result['causal_mask_applied']}")
                if 'final_output' in result:
                    output_shape = np.array(result['final_output']).shape if result['final_output'] else "None"
                    logger.info(f"Run {run_id} - Final output shape: {output_shape}")
            except Exception as e:
                logger.warning(f"Run {run_id} - Error logging result details: {e}")
        
        if success:
            logger.info(f"✓ Run {run_id}: {message}")
            print(f"✓ Run {run_id}: {message}")
        else:
            logger.warning(f"✗ Run {run_id}: {message}")
            print(f"✗ Run {run_id}: {message}")

        return run_id, success, result, message
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"Exception during execution: {str(e)}"
        logger.error(f"Run {run_id} failed after {execution_time:.2f}s - {error_msg}")
        print(f"✗ Run {run_id}: {error_msg}")
        return run_id, False, None, error_msg


async def main(concurrent: bool = True, verbose_runs: bool = False):
    tools: list[ToolUnionParam] = [
        {
            "name": "python_expression",
            "description": "Evaluates Python expressions. Use print() to output results. Has numpy imported as 'np'.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Python code to execute. Can span multiple lines.",
                    }
                },
                "required": ["expression"],
            },
        },
        {
            "name": "submit_answer",
            "description": "Submit the final answer as a dictionary with keys: final_x, final_y, converged",
            "input_schema": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "object",
                        "description": "Dictionary with final_x, final_y (final coordinates), and converged (boolean)",
                    }
                },
                "required": ["answer"],
            },
        },
    ]

    tool_handlers = {
        "python_expression": python_expression_tool,
        "submit_answer": submit_answer_tool,
    }

    # The task prompt - implementing gradient descent with momentum
    prompt = """You are an ML engineer implementing a multi-head attention mechanism with position encoding.

**TASK**: Implement "Positional Multi-Head Attention" - attention with learnable position embeddings and multiple heads.

**ALGORITHM**:
Given query Q, key K, value V matrices (each 4x3), compute 2-head attention with positional encoding:

1. **Position Encoding**: 
   - Add learnable position embeddings to inputs
   - pos_emb = [[0.1, 0.0, -0.1], [0.0, 0.1, 0.0], [-0.1, 0.0, 0.1], [0.0, -0.1, 0.0]]  # 4x3
   - Q_pos = Q + pos_emb, K_pos = K + pos_emb, V_pos = V + pos_emb

2. **Multi-Head Projection**:
   - Split each 3D vector into 2 heads of dimension 1 and 2
   - For head h1: Q1[i] = [Q_pos[i][0]], K1[i] = [K_pos[i][0]], V1[i] = Q_pos[i][:2]  # dim 1 for Q,K; dim 2 for V
   - For head h2: Q2[i] = Q_pos[i][1:], K2[i] = K_pos[i][1:], V2[i] = [Q_pos[i][2]]  # dim 2 for Q,K; dim 1 for V

3. **Per-Head Attention**:
   - For each head h: A_h[i,j] = Q_h[i] · K_h[j] / sqrt(d_k)  # d_k = dimension of keys
   - Apply mask: zero out A_h[i,j] where i > j (causal mask)
   - W_h[i,j] = softmax(A_h[i,:])
   - O_h[i] = Σⱼ W_h[i,j] * V_h[j]

4. **Head Concatenation**:
   - Concatenate outputs: O[i] = [O1[i], O2[i]]  # Should result in 4x3 matrix
   - Handle dimension mismatch by padding O2 with zeros to match

**INPUT DATA**:
```python
Q = [[1.0, 0.5, -0.2], [0.3, -0.1, 0.8], [0.0, 0.4, -0.5], [-0.2, 0.2, 0.1]]
K = [[0.8, 0.2, -0.1], [0.1, -0.3, 0.6], [0.4, 0.7, 0.0], [0.0, 0.0, 0.9]]  
V = [[1.5, -0.8, 0.3], [0.2, 1.0, -0.4], [-0.1, 0.6, 1.2], [0.9, 0.0, -0.7]]
```

**IMPLEMENTATION CHALLENGES**:
- Position embedding addition
- Multi-head dimension splitting and management
- Causal masking (upper triangular zeroing)
- Different dimensions per head for Q, K, V
- Scaled dot-product attention with sqrt(d_k)
- Proper concatenation with dimension handling
- Numerical stability in softmax

**OUTPUT**: Submit a dictionary with:
- "position_encoded_queries": Q_pos matrix (4x3) after adding position embeddings
- "head1_attention_weights": W1 matrix (4x4) for first head
- "head2_attention_weights": W2 matrix (4x4) for second head  
- "head1_output": O1 matrix (4x2) output from first head
- "head2_output": O2 matrix (4x1) output from second head
- "final_output": O matrix (4x3) concatenated result
- "causal_mask_applied": boolean indicating if you applied causal masking correctly

Use the python_expression tool to implement this step by step, then submit_answer with the required format."""

    # Run the test multiple times to measure pass rate
    num_runs = 15
    execution_mode = "concurrently" if concurrent else "sequentially"
    print(f"Running {num_runs} test iterations {execution_mode}...")
    print("=" * 80)

    tasks = [
        run_single_test(
            run_id=i + 1,
            num_runs=num_runs,
            prompt=prompt,
            tools=tools,
            tool_handlers=tool_handlers,
            verbose=verbose_runs,
        )
        for i in range(num_runs)
    ]

    if concurrent:
        results = []
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
    else:
        results = []
        for task in tasks:
            result = await task
            results.append(result)

    # Analyze results
    successes = sum(success for _, success, _, _ in results)
    pass_rate = (successes / num_runs) * 100

    print(f"\n{'=' * 80}")
    print("TASK ANALYSIS - Positional Multi-Head Attention Implementation")
    print(f"{'=' * 80}")
    print(f"Total Runs: {num_runs}")
    print(f"Successful: {successes}")
    print(f"Failed: {num_runs - successes}")
    print(f"Pass Rate: {pass_rate:.1f}%")
    
    # Analyze failure modes
    failure_reasons = {}
    for run_id, success, result, message in results:
        if not success:
            # Categorize failure type
            if "No answer submitted" in message:
                reason = "No submission"
            elif "Missing required keys" in message:
                reason = "Invalid format"
            elif "not converged" in message:
                reason = "Convergence detection"
            elif "distance from optimum" in message:
                reason = "Incorrect optimization"
            elif "Error parsing" in message:
                reason = "Parse error"
            else:
                reason = "Other"
            
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

    if failure_reasons:
        print(f"\nFailure Analysis:")
        for reason, count in failure_reasons.items():
            print(f"  {reason}: {count} runs ({count/num_runs*100:.1f}%)")

    print(f"{'=' * 80}")
    
    # Check if pass rate is in target range (10-40%)
    if 10 <= pass_rate <= 40:
        print("✓ PASS RATE IN TARGET RANGE (10-40%)")
    else:
        print(f"⚠ PASS RATE OUTSIDE TARGET RANGE: {pass_rate:.1f}% (target: 10-40%)")


if __name__ == "__main__":
    # Set verbose_runs=True to see detailed execution of individual runs
    asyncio.run(main(concurrent=True, verbose_runs=False))