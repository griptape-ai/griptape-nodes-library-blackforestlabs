# FLUX Klein Model Support - Feature Summary

## Overview
Successfully added support for FLUX 2 Klein models to the `TextToImage` node in the Black Forest Labs library.

## Branch
`feature/add-flux-klein-support`

## Changes Made

### 1. Code Changes (`text_to_image.py`)

#### Model Options
- Added `flux-2-klein-4b` (Apache 2.0 license, ~13GB VRAM)
- Added `flux-2-klein-9b` (FLUX NCL license, ~24GB VRAM)
- Updated model tooltip to highlight Klein's sub-second generation speed

#### UI/UX Enhancements
- **Dynamic Parameter Visibility**: Klein models don't support `prompt_upsampling`, so this parameter is now hidden when a Klein model is selected
- **Initialization Method**: Added `_initialize_parameter_visibility()` to set correct UI state on node creation
- **Runtime Updates**: Modified `after_value_set()` to show/hide `prompt_upsampling` when model selection changes

#### Technical Implementation
- Klein models use the same API pattern as classic FLUX models (width/height parameters)
- Automatic reset of `prompt_upsampling` to `False` when switching to Klein
- Proper handling of Kontext vs FLUX vs Klein model families

### 2. Documentation Updates (`README.md`)

#### Features Section
- Updated to highlight Klein models' sub-second generation capability
- Added licensing information (Apache 2.0 for 4B, FLUX NCL for 9B)

#### Available Nodes Section
- Added Klein model options with VRAM requirements
- Clear indication of each model's characteristics

## Implementation Details

### Best Practices Followed
✅ **Parameter Initialization Pattern**: Initialize UI visibility on node creation  
✅ **Dynamic UI Updates**: Use `after_value_set()` for responsive parameter visibility  
✅ **Clear Naming**: Method names clearly indicate their purpose  
✅ **Comments**: Inline comments explain Klein-specific behavior  
✅ **Comprehensive Documentation**: Updated README with all relevant information  
✅ **Conventional Commits**: Used semantic commit messages

### Code Structure
```python
def _initialize_parameter_visibility(self) -> None:
    """Initialize parameter visibility based on default model."""
    default_model = self.get_parameter_value("model") or "flux-pro-1.1"
    if isinstance(default_model, str) and default_model.startswith("flux-2-klein"):
        self.hide_parameter_by_name("prompt_upsampling")

def after_value_set(self, parameter: Parameter, value: Any) -> None:
    # ... existing aspect_ratio/max_size logic ...
    
    # Hide prompt_upsampling for Klein models (they don't support it)
    if parameter.name == "model":
        if isinstance(value, str) and value.startswith("flux-2-klein"):
            self.hide_parameter_by_name("prompt_upsampling")
            self.set_parameter_value("prompt_upsampling", False)
        else:
            self.show_parameter_by_name("prompt_upsampling")
```

## FLUX 2 Klein Models

### Key Features
- **Sub-second inference**: Fastest FLUX models available
- **Open weights**: Community can use and modify
- **Consumer GPU friendly**: Runs on standard hardware

### Model Variants
| Model | License | VRAM | Speed | API Pricing |
|-------|---------|------|-------|-------------|
| flux-2-klein-4b | Apache 2.0 | ~13GB | Sub-second | $0.014 + $0.001/MP |
| flux-2-klein-9b | FLUX NCL | ~24GB | Sub-second | $0.015 + $0.002/MP |

### Limitations
- **No prompt upsampling**: Klein models don't support this feature
- **Max 4 input images**: Lower than other FLUX models (which support 8-10)
- **Up to 4MP resolution**: Same as other FLUX models

## Testing Recommendations

### Test Cases
1. **Model Selection**
   - Select Klein 4B → verify `prompt_upsampling` is hidden
   - Select Klein 9B → verify `prompt_upsampling` is hidden
   - Select Pro/Ultra/Dev → verify `prompt_upsampling` is visible
   - Switch between Klein and non-Klein → verify parameter visibility updates

2. **Image Generation**
   - Generate image with Klein 4B model
   - Generate image with Klein 9B model
   - Verify generation works with all aspect ratios
   - Verify seed reproducibility

3. **API Integration**
   - Verify correct API endpoint usage
   - Verify payload structure matches Klein requirements
   - Verify polling mechanism works correctly

## Commits

```
f51ac13 docs: update README with FLUX Klein model information
267d1d2 feat: add FLUX Klein model support to TextToImage node
```

## Next Steps

1. **Test the changes** in a live Griptape Nodes environment
2. **Create a pull request** to merge into main branch
3. **Update version number** if needed (following semantic versioning)
4. **Consider adding Klein models** to other nodes that support FLUX models

## References

- [FLUX 2 Klein Documentation](https://docs.bfl.ai/flux_2/flux2_overview#flux-2-%5Bklein%5D-models)
- [Griptape Node Development Guide v3](../griptape-nodes-node-development-guide/node-development-guide-v3.md)
