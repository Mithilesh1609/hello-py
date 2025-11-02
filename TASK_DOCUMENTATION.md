# Positional Multi-Head Attention RL Task

## Overview

This repository contains a reinforcement learning training task designed for LLM agents. The task implements a **Positional Multi-Head Attention mechanism** - a core component of transformer architectures used in production ML systems like GPT, BERT, and other state-of-the-art models.

**Target Audience**: ML Engineers, AI Researchers, RL Training Pipeline Developers

**Difficulty Level**: Intermediate to Advanced (Target pass rate: 10-40%)

---

## 🎯 Task Objectives

### Primary Goals
- **Educational**: Teach LLMs fundamental transformer components through hands-on implementation
- **Practical**: Simulate real ML engineering work involving attention mechanisms
- **Assessment**: Evaluate LLM capability on multi-step algorithmic reasoning
- **Balanced Difficulty**: Maintain 10-40% success rate for effective RL training

### Skills Assessed
- Matrix operations and linear algebra
- Attention mechanism understanding  
- Position encoding concepts
- Multi-head architecture design
- Causal masking for autoregressive models
- Numerical implementation precision
- Debugging and error handling

---

## 📋 Problem Statement

### Task Description
Implement a **Positional Multi-Head Attention mechanism** from scratch given Query (Q), Key (K), and Value (V) matrices.

### Input Data
```python
Q = [[1.0, 0.5, -0.2], [0.3, -0.1, 0.8], [0.0, 0.4, -0.5], [-0.2, 0.2, 0.1]]  # 4x3
K = [[0.8, 0.2, -0.1], [0.1, -0.3, 0.6], [0.4, 0.7, 0.0], [0.0, 0.0, 0.9]]   # 4x3  
V = [[1.5, -0.8, 0.3], [0.2, 1.0, -0.4], [-0.1, 0.6, 1.2], [0.9, 0.0, -0.7]] # 4x3
```

### Algorithm Requirements

#### Step 1: Position Encoding
Add learnable position embeddings to input matrices:
```python
pos_emb = [[0.1, 0.0, -0.1], [0.0, 0.1, 0.0], [-0.1, 0.0, 0.1], [0.0, -0.1, 0.0]]
Q_pos = Q + pos_emb
K_pos = K + pos_emb  
V_pos = V + pos_emb
```

#### Step 2: Multi-Head Projection
Split each 3D vector into 2 heads with **asymmetric dimensions**:
- **Head 1**: Q,K dimension=1, V dimension=2
- **Head 2**: Q,K dimension=2, V dimension=1

```python
# Head 1
Q1[i] = [Q_pos[i][0]]      # 1D
K1[i] = [K_pos[i][0]]      # 1D  
V1[i] = Q_pos[i][:2]       # 2D

# Head 2  
Q2[i] = Q_pos[i][1:]       # 2D
K2[i] = K_pos[i][1:]       # 2D
V2[i] = [Q_pos[i][2]]      # 1D
```

#### Step 3: Per-Head Attention
For each head, compute scaled dot-product attention:
```python
# Attention scores
A_h[i,j] = Q_h[i] · K_h[j] / sqrt(d_k)

# Causal masking (critical requirement)
if i > j: A_h[i,j] = -inf

# Softmax normalization
W_h[i,j] = softmax(A_h[i,:])

# Weighted sum
O_h[i] = Σⱼ W_h[i,j] * V_h[j]
```

#### Step 4: Output Concatenation
Combine head outputs into final 4x3 matrix:
```python
O_final = concatenate([O1, O2], axis=1)  # (4,2) + (4,1) → (4,3)
```

---

## 📤 Expected Output Format

The LLM must submit a dictionary with exactly these keys:

```python
{
    "position_encoded_queries": Q_pos,        # 4x3 matrix
    "head1_attention_weights": W1,            # 4x4 matrix
    "head2_attention_weights": W2,            # 4x4 matrix  
    "head1_output": O1,                       # 4x2 matrix
    "head2_output": O2,                       # 4x1 matrix
    "final_output": O_final,                  # 4x3 matrix
    "causal_mask_applied": True               # Boolean flag
}
```

---

## ✅ Success & Failure Criteria

### Tolerance-Based Grading System

The verification system uses graduated tolerance levels to assess different complexity levels:

| Component | Tolerance | Rationale |
|-----------|-----------|-----------|
| Position Encoding | 0.1 | Fundamental concept, should be precise |
| Attention Weights | 0.15 | Core mechanism, moderate tolerance |
| Output Values | 0.2 | Final results, more lenient |

### Success Conditions (OR Logic)

#### **Full Success**
- ✅ Position encoding correct (`max_error < 0.1`)
- ✅ Both attention heads correct (`max_error < 0.15`)
- ✅ Causal mask applied (`causal_mask_applied = True`)
- ✅ At least one head output correct (`max_error < 0.2`)
- ✅ Final concatenation correct (`max_error < 0.2`)

**Success Message**: `"SUCCESS: Complete multi-head attention. Errors: pos=X.XXX, w1=X.XXX, w2=X.XXX, final=X.XXX"`

#### **Partial Success**
- ✅ Position encoding correct
- ✅ At least ONE attention head correct  
- ✅ Causal mask applied
- ❌ May have issues with output/concatenation

**Success Message**: `"SUCCESS: Core concepts correct, one head working. Errors: pos=X.XXX, better_head=X.XXX"`

### Failure Conditions

####  **Automatic Failures**
1. **Format Errors**: Missing required keys, wrong data types
2. **Shape Errors**: Incorrect matrix dimensions
3. **No Submission**: Agent doesn't submit answer within 15 steps
4. **Parse Errors**: Invalid JSON or data conversion issues

#### **Algorithmic Failures**
1. **Missing Causal Mask**: `causal_mask_applied = False`
2. **Incorrect Position Encoding**: `error ≥ 0.1`
3. **Both Attention Heads Wrong**: `error ≥ 0.15` for both heads
4. **Output/Concatenation Errors**: `error ≥ 0.2`

**Failure Message Examples**:
- `"PARTIAL: Good position encoding but missing causal mask. Pos_err: X.XXX"`
- `"FAILURE: Incorrect position_encoding (err: X.XXX), head1_weights (err: X.XXX), causal_mask_missing"`

---

## Technical Architecture

### Framework Components

#### 1. **Agent Interaction System**
- **LLM Client**: AsyncAnthropic with Claude Sonnet 4.5
- **Tool Interface**: `python_expression_tool` and `submit_answer_tool`
- **Max Steps**: 15 iterations per attempt
- **Timeout Handling**: Graceful failure on non-submission

#### 2. **Verification Engine**
```python
def verify_gradient_descent_solution(result: dict) -> tuple[bool, str]:
    """
    Validates submitted solution against reference implementation
    Returns: (success: bool, detailed_message: str)
    """
```

**Reference Implementation**: Provides ground truth for numerical comparison

#### 3. **Concurrent Testing Framework**
```python
async def run_single_test(...) -> tuple[int, bool, Any, str]:
    """
    Executes single test run with comprehensive logging
    Returns: (run_id, success, result, message)
    """
```

**Features**:
- Async execution for performance
- Individual run tracking and timing
- Detailed error categorization
- Comprehensive logging system

#### 4. **Logging Infrastructure**
```python
# Timestamped file logging + console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'gradient_descent_task_{timestamp}.log'),
        logging.StreamHandler()
    ]
)
```

**Logged Information**:
- Individual run start/completion times
- Success/failure outcomes with detailed error messages
- Submitted answer structure and key metrics
- Shape validation and dimension checks
- Causal mask application status
- Component-wise error measurements

---

## 📊 Performance Metrics & Analysis

### Target Performance
- **Pass Rate**: 10-40% (optimal for RL training)
- **Execution Time**: ~15-30 seconds per run
- **Concurrent Runs**: 15 parallel tests for statistical significance

### Common Failure Modes

Based on empirical testing, typical failure patterns include:

1. **Dimension Management** (30-40% of failures)
   - Incorrect head splitting logic
   - Shape mismatches in matrix operations
   - Concatenation dimension errors

2. **Causal Masking** (25-35% of failures)
   - Forgetting to apply mask entirely
   - Incorrect masking direction (i < j vs i > j)
   - Improper -inf assignment

3. **Numerical Stability** (15-25% of failures)
   - Softmax overflow/underflow
   - Missing scaling factor (√d_k)
   - Precision errors in floating-point operations

4. **Implementation Logic** (10-20% of failures)
   - Incorrect position embedding addition
   - Wrong attention computation order
   - Algorithm misunderstanding

### Performance Analysis Output
```bash
================================================================================
TASK ANALYSIS - Positional Multi-Head Attention Implementation
================================================================================
Total Runs: 15
Successful: 4
Failed: 11
Pass Rate: 26.7%

Failure Analysis:
  Dimension errors: 5 runs (33.3%)
  Missing causal mask: 3 runs (20.0%)
  Numerical issues: 2 runs (13.3%)
  No submission: 1 run (6.7%)

✓ PASS RATE IN TARGET RANGE (10-40%)
================================================================================
```

---

## 🚀 Usage Instructions

### Prerequisites
```bash
# Install dependencies
uv add anthropic numpy

# Set up Anthropic API key
export ANTHROPIC_API_KEY="your-key-here"
```

### Running the Task
```bash
# Standard execution (15 concurrent runs)
cd /path/to/hello-py
uv run gradient_descent_task.py

# With verbose individual run details
# Edit main() call: asyncio.run(main(concurrent=True, verbose_runs=True))
```

### Configuration Options
```python
# In main() function
num_runs = 15              # Number of test iterations
concurrent = True          # Run tests in parallel
verbose_runs = False       # Show detailed run execution
model = "claude-sonnet-4-5-20250929"  # LLM model
```

### Log File Analysis
Log files are automatically generated with timestamps:
```
gradient_descent_task_20251102_143025.log
```

**Key log patterns to monitor**:
- Run completion times (performance)
- Success/failure ratios (difficulty balance)
- Common error patterns (task refinement)
- Shape validation issues (implementation bugs)

---

## 🔧 Development & Customization

### Modifying Difficulty

#### Increase Difficulty (Lower Pass Rate)
1. **Tighter Tolerances**: Reduce tolerance values
2. **Additional Requirements**: Add more heads or complex masking
3. **Stricter Success Criteria**: Require all components to be correct

#### Decrease Difficulty (Higher Pass Rate)
1. **Looser Tolerances**: Increase tolerance values
2. **Partial Credit**: Allow more partial success conditions
3. **Simplified Algorithm**: Remove position encoding or use single head

### Adding New Components

#### Example: Layer Normalization
```python
# Add to algorithm steps
def apply_layer_norm(x):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + 1e-8)

# Add to output requirements
"layer_normalized_output": layer_norm_result
```

#### Example: Residual Connections
```python
# Modify final output computation
O_final = input_embeddings + concatenate([O1, O2], axis=1)
```

### Custom Verification Logic
```python
def verify_custom_solution(result: dict) -> tuple[bool, str]:
    """
    Custom verification for modified tasks
    """
    # Add your custom validation logic here
    return success, message
```

---

## 🧪 Testing & Validation

### Unit Testing the Verification System
```python
# Test reference implementation
def test_reference_implementation():
    ref_result = reference_implementation()
    success, message = verify_gradient_descent_solution(ref_result)
    assert success, f"Reference implementation failed: {message}"

# Test tolerance boundaries
def test_tolerance_boundaries():
    # Test cases at tolerance edges
    pass
```

### Benchmarking Performance
```python
# Measure execution time distribution
execution_times = []
for run_result in results:
    execution_times.append(run_result.execution_time)

print(f"Avg execution time: {np.mean(execution_times):.2f}s")
print(f"95th percentile: {np.percentile(execution_times, 95):.2f}s")
```

### Validating Pass Rate Stability
```python
# Run multiple batches to ensure consistent pass rates
batch_results = []
for batch in range(10):
    batch_pass_rate = run_test_batch(num_runs=15)
    batch_results.append(batch_pass_rate)

print(f"Pass rate stability: {np.std(batch_results):.1f}% std dev")
```

---

## 📚 References & Further Reading

### Transformer Architecture Papers
- "Attention Is All You Need" (Vaswani et al., 2017)
- "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
- "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)

### Multi-Head Attention Resources
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Understanding Multi-Head Attention](https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853)
- [Attention Mechanisms in Neural Networks](https://distill.pub/2016/augmented-rnns/)

### RL Training for LLMs
- "Training language models to follow instructions with human feedback" (Ouyang et al., 2022)
- "Constitutional AI: Harmlessness from AI Feedback" (Bai et al., 2022)

---

## 🤝 Contributing

### Adding New Tasks
1. Implement verification function following the tolerance-based pattern
2. Design algorithm with 10-40% target difficulty
3. Add comprehensive logging and error handling
4. Test pass rate stability across multiple runs
5. Update documentation with task details

### Code Style Guidelines
- Use type hints for all function parameters
- Add docstrings for public functions
- Follow numpy array manipulation best practices
- Include error handling for edge cases
- Maintain consistent logging format

### Reporting Issues
When reporting bugs or suggesting improvements:
1. Include pass rate statistics
2. Provide log file excerpts
3. Describe expected vs actual behavior
4. Suggest specific tolerance adjustments if applicable

---

## 📄 License

This project is part of the AIChamp RL training pipeline. See repository license for usage terms.

---

**Last Updated**: November 2, 2025  
**Version**: 1.0  
**Maintainer**: RL Training Team