import Sensors as s 

# Import classifiers
import TableClassifier as table
import SteadyStateClassifier as ss 
# import TheftClassifier as theft
import HandClassifier as hand
import PBBvsAllClassifier as pbb

### CLASSIFIERS ###
TABLE_CLASSIFIER = "Table Classifier"
POCKET_BAG_CLASSIFIER = "Pocket/Bag Classifier"
THEFT_CLASSIFIER = "Theft Classifier"
HAND_CLASSIFIER = "Hand Classifier"

RELEVANT_SENSORS = [s.ACCELEROMETER, s.STEP_COUNT]
CLASSIFIERS = {
    TABLE_CLASSIFIER: table.Classifier(),
    # POCKET_BAG_CLASSIFIER: pbb.Classifier(),
    # THEFT_CLASSIFIER: theft.Classifier()
    HAND_CLASSIFIER: hand.Classifier()
}

FILE_INTERVALS = "fileIntervals"
DAY_INTERVAL = "dayIntervals"