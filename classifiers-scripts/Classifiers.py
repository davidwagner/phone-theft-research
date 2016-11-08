import Sensors as s 

# Import classifiers
import TableClassifier as table

### CLASSIFIERS ###
RELEVANT_SENSORS = [s.ACCELEROMETER, s.STEP_COUNT]
CLASSIFIERS = {
    "tableClassifier": table.Classifier()
}