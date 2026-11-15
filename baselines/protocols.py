"""External repetition policies; our method is never assigned extra seeds."""

def seeds_for(method, evaluation, configured_seed=42):
    # ccIDIR §IV-B: 50 independent fits per DIR-Lab case, both directions
    # from each fitted pair. Published numeric seed IDs are not supplied.
    if method == 'ccidir' and evaluation == 'dirlab':
        return list(range(50))
    return [configured_seed]
