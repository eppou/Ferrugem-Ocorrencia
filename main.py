import time
from datetime import datetime

# import result.train_test_model as tm
import result.test_severity_model as smt

if __name__ == "__main__":
    print(datetime.now().strftime("Execution started at %Y-%m-%d %I:%M:%S %p"))
    print()
    start = time.time()

    smt.run()

    end = time.time()

    print()
    print(datetime.now().strftime("Execution ended at %Y-%m-%d %I:%M:%S %p"))
    print(f"Execution took {round((end - start) * 10**3)} ms")
