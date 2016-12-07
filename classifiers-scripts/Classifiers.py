import Sensors as s 

# Import classifiers
# import TableClassifier as table
import SteadyStateClassifier as ss 
import TableClassifier as table
import theft_classifier as theft

### CLASSIFIERS ###
TABLE_CLASSIFIER = "Table Classifier"
POCKET_BAG_CLASSIFIER = "Pocket/Bag Classifier"
THEFT_CLASSIFIER = "Theft Classifier"

RELEVANT_SENSORS = [s.ACCELEROMETER, s.STEP_COUNT]
CLASSIFIERS = {
    TABLE_CLASSIFIER: table.Classifier(),
    POCKET_BAG_CLASSIFIER: ss.Classifier(),
    # "theftClassifier" : theft.Classifier()
}

FILE_INTERVALS = "fileIntervals"
DAY_INTERVAL = "dayIntervals"