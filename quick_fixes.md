# 🔧 Quick Fixes for Sherlock Issues

## 1. **Mass Ratio Fix - Kroupa IMF**

**Problem**: Mass ratio 0.005 instead of ~1.0 due to 1M sample cap

**Root Cause**: In `kroupa_imf.py` line 91:
```python
n_poisson = min(n_poisson, max_samples)  # Safety cap at 1M
```

**Fix**: Increase `max_samples` or make it adaptive in `train_tfp_flows.py`:

```python
# In train_tfp_flows.py, around line 364:
samples, masses = sample_with_kroupa_imf(
    flow=flow,
    n_target_mass=stellar_mass,
    preprocessing_stats=preprocessing_stats,
    max_samples=10_000_000,  # Increase from 1M to 10M
    seed=42
)
```

## 2. **Filter Script Fix**

**Problem**: `filter_completed_particles.sh` failing

**Quick Debug**:
```bash
# On Sherlock, check:
cd /oak/stanford/orgs/kipac/users/caganze/flows-tensorflow
ls -la filter_completed_particles.sh
./filter_completed_particles.sh --help
./filter_completed_particles.sh --dry-run --verbose
```

**Likely Issues**:
- File paths not matching expected structure
- Permissions issue
- Missing particle_list.txt

## 3. **Quick Test Command**

**To test the mass ratio fix on Sherlock**:
```bash
# Edit train_tfp_flows.py to increase max_samples
# Then test with a small particle:
python train_tfp_flows.py \
    --data_path /path/to/small/halo.h5 \
    --particle_pid 1 \
    --output_dir test_output \
    --epochs 2 \
    --batch_size 128 \
    --generate-samples \
    --n_samples 100000
```

## 4. **Expected Results After Fix**

**Before**: Mass ratio ~0.005  
**After**: Mass ratio ~1.0 (between 0.5-2.0)

The Kroupa IMF distribution will remain correct, but the total mass will match the target stellar mass.

