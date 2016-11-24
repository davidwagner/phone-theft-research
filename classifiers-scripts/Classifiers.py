import Sensors as s 

# Import classifiers
# import TableClassifier as table
import SteadyStateClassifier as ss 

### CLASSIFIERS ###
RELEVANT_SENSORS = [s.ACCELEROMETER, s.STEP_COUNT]
CLASSIFIERS = {
    # "tableClassifier": table.Classifier()
    "steadyStateClassifier": ss.Classifier()
}