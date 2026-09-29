"""Keep basic flights for retention checks without reinforcement gradients."""
def reinforcement_coefficient(entry):
    kind=entry.get('natural')
    if kind=='basic':return 0.0
    if kind is None or kind=='full':return 1.0
    raise ValueError('Unknown training task')
