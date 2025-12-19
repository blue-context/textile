# Temporal Transformers

Temporal transformers filter or modify messages based on their age in the conversation.

## DecayTransformer

**Pattern**: Exponential decay based on message age

### Formula
```
new_prominence = old_prominence * 0.5^(age_turns / half_life)
```

### When to Use
- Long conversations (10+ turns)
- Recent context is more important than old context
- Gradual forgetting is acceptable
- Token budget needs management

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `half_life_turns` | 5 | Turns for prominence to decay by 50% |
| `threshold` | 0.1 | Min prominence to keep (0.0-1.0) |
| `min_recent_messages` | 10 | Always keep N recent messages |

### Example Usage

```python
import textile
from examples.reference_transformers.temporal.decay import DecayTransformer

# Basic usage: aggressive pruning
textile.configure(
    transformers=[
        DecayTransformer(
            half_life_turns=3,    # Fast decay
            threshold=0.2,         # Keep only prominent
            min_recent_messages=5  # Keep at least 5 recent
        )
    ]
)

# Conservative: keep more history
textile.configure(
    transformers=[
        DecayTransformer(
            half_life_turns=10,   # Slow decay
            threshold=0.05,        # Keep almost everything
            min_recent_messages=20 # Large safety net
        )
    ]
)
```

### Customization Ideas

1. **Linear Decay**: Replace exponential with linear
   ```python
   decay_factor = max(0, 1 - (age_turns / max_age))
   ```

2. **Role-Specific Decay**: Different rates per role
   ```python
   if msg.role == "system":
       decay_factor = 1.0  # Never decay
   elif msg.role == "user":
       decay_factor = 0.5 ** (age / user_half_life)
   else:
       decay_factor = 0.5 ** (age / assistant_half_life)
   ```

3. **Stepped Decay**: Discrete decay tiers
   ```python
   if age_turns < 5:
       decay_factor = 1.0
   elif age_turns < 10:
       decay_factor = 0.5
   else:
       decay_factor = 0.1
   ```

### Performance

- **Computational Cost**: O(n) where n = message count
- **Typical Latency**: < 1ms for 100 messages
- **Memory**: Negligible (in-place metadata updates)

### Edge Cases Handled

✅ Empty context → No-op
✅ Single message → Not applied (see `should_apply`)
✅ All messages filtered → Keeps best message
✅ System messages → Always preserved
✅ Recent messages → Minimum guarantee

### Testing

See `decay_test.py` for comprehensive test suite covering:
- Basic decay calculation
- Threshold filtering
- Minimum recent messages guarantee
- System message preservation
- Edge cases

### Related Patterns

- **SemanticDecayTransformer**: Combines temporal + semantic
- **Token Budget Transformer**: Hard limit enforcement
- **Adaptive Decay**: Adjust half-life based on conversation
