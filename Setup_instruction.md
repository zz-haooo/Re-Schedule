## Path Configuration

All scripts now use relative paths and environment variables. Key changes:

- **Model paths**: Use `$MODEL_PATH` environment variable
- **Dataset paths**: Relative to project root (`./datasets/`)
- **Output paths**: Relative to project root (`./outputs/`, `./rollout_data/`)
- **Python paths**: Automatically resolved using script directory

## File Structure

```
Re-Schedule/
├── datasets/           # Training and evaluation datasets
├── reasoning_tree/     # Tree-based reasoning evaluation tools
├── run/               # Training scripts and outputs
├── verl/              # Modified VERL framework
├── .gitignore         # Git ignore rules
├── README.md          # Main documentation
├── config_example.txt # Environment configuration example
└── SETUP_INSTRUCTIONS.md # This file
```
