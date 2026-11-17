"""Classic SimpleITK Demons with the official DemonsRegistration1 recipe."""
import time
from .common import finish, preflight, predict_for_case
from .sources import verify_source, native_modules
from .physical import sitk_image, lps_displacement_predictor, provenance_checkpoint


def run_case(case, config, device):
    preflight(case, config, 'demons')
    verify_source('demons')
    module, = native_modules('demons')
    started = time.monotonic()
    fixed, moving = (sitk_image(module, case[k]) for k in ('fixed', 'moving'))
    options = config['native']
    if options['histogram_matching']:
        matcher = module.HistogramMatchingImageFilter()
        matcher.SetNumberOfWorkUnits(4)
        matcher.SetNumberOfHistogramLevels(options['histogram_levels'])
        matcher.SetNumberOfMatchPoints(options['match_points'])
        matcher.ThresholdAtMeanIntensityOn()
        moving = matcher.Execute(moving, fixed)
    registration = module.DemonsRegistrationFilter()
    registration.SetNumberOfWorkUnits(4)
    registration.SetNumberOfIterations(options['iterations'])
    registration.SetStandardDeviations(options['standard_deviations'])
    fit_started = time.monotonic()
    result = registration.Execute(fixed, moving)
    registration_seconds = time.monotonic() - fit_started
    field = module.GetArrayFromImage(result).transpose(2, 1, 0, 3)
    predict = lps_displacement_predictor(field, case)
    return finish('demons', case, config, predict_for_case(case, config, predict), provenance_checkpoint(), started,
                  variant='classic Demons, official SimpleITK example', backend='CPU',
                  registration_seconds=registration_seconds,
                  elapsed_iterations=int(registration.GetElapsedIterations()), native_metric=float(registration.GetMetric()),
                  preprocessing='MRI histogram matching' if options['histogram_matching'] else 'native CT intensities',
                  optimization_roi='whole_image')
