# Multi-Head Attention RL Training Task

This repository contains a reinforcement learning training task for LLM agents implementing **Positional Multi-Head Attention** - a core transformer component.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/preferencemodel/hello-py.git
cd hello-py

# Install dependencies
uv add anthropic numpy

# Set up API key
export ANTHROPIC_API_KEY="your-key-here"

# Run the task
uv run gradient_descent_task.py
```

## 📋 Task Overview

- **Task**: Implement multi-head attention with position encoding from scratch
- **Difficulty**: Intermediate to Advanced (10-40% target pass rate)
- **Skills**: Matrix operations, attention mechanisms, causal masking
- **Output**: Comprehensive performance analysis with detailed logging

## 📚 Complete Documentation

**👉 See [TASK_DOCUMENTATION.md](./TASK_DOCUMENTATION.md) for full technical details including:**

- Complete problem statement and algorithm requirements
- Success/failure criteria and tolerance specifications  
- Technical architecture and logging infrastructure
- Performance metrics and common failure modes
- Development guidelines and customization options
- Testing procedures and validation methods

## 🎯 Target Metrics

- **Pass Rate**: 10-40% (optimal for RL training)
- **Execution**: 15 concurrent test runs
- **Timing**: ~15-30 seconds per run
- **Components**: Position encoding, multi-head splitting, causal masking, attention computation

## 📊 Sample Output

```
================================================================================
TASK ANALYSIS - Positional Multi-Head Attention Implementation  
================================================================================
Total Runs: 15
Successful: 4
Failed: 11
Pass Rate: 26.7%

✓ PASS RATE IN TARGET RANGE (10-40%)
================================================================================
```

## 🔧 Configuration Options

Edit the execution parameters in `gradient_descent_task.py`:

```python
# At bottom of file
asyncio.run(main(
    concurrent=True,      # Run tests in parallel
    verbose_runs=False    # Show detailed run execution
))
```

## Legacy Files

- `main.py`: Original gradient descent implementation
- This project has evolved to focus on the multi-head attention task in `gradient_descent_task.py`

For developers working on RL training pipelines, this task provides a balanced challenge that teaches fundamental ML engineering skills while maintaining appropriate difficulty for effective learning.
