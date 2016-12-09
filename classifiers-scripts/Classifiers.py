import Sensors as s 

# Import classifiers
# import TableClassifier as table
# import SteadyStateClassifier as ss 
# import TableClassifier as table
import TheftClassifier as theft

### CLASSIFIERS ###
TABLE_CLASSIFIER = "Table Classifier"
POCKET_BAG_CLASSIFIER = "Pocket/Bag Classifier"
THEFT_CLASSIFIER = "Theft Classifier"

RELEVANT_SENSORS = [s.ACCELEROMETER, s.STEP_COUNT]
CLASSIFIERS = {
    TABLE_CLASSIFIER: table.Classifier(),
    POCKET_BAG_CLASSIFIER: ss.Classifier(),
    # THEFT_CLASSIFIER: theft.Classifier()
}

FILE_INTERVALS = "fileIntervals"
DAY_INTERVAL = "dayIntervals"