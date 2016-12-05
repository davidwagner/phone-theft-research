import Sensors as s 

# Import classifiers
# import TableClassifier as table
import SteadyStateClassifier as ss 
import TableClassifier as table
import theft_classifier as theft

### CLASSIFIERS ###
RELEVANT_SENSORS = [s.ACCELEROMETER, s.STEP_COUNT]
CLASSIFIERS = {
    "tableClassifier": table.Classifier(),
    "steadyStateClassifier": ss.Classifier(),
    "theftClassifier" : theft.Classifier()
}

FILE_INTERVALS = "fileIntervals"
DAY_INTERVALS = "dayIntervals"