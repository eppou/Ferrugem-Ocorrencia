import time
from datetime import datetime

import procedures.prepare_ferrugem_ocorrencia_instances as prepare_instances
import procedures.prepare_ferrugem_ocorrencia_features as prepare_features
import procedures.train_test_model as tm
import procedures.prepare_severity_model as sm
import procedures.test_severity_model as smt

if __name__ == "__main__":
    print(datetime.now().strftime("Execution started at %Y-%m-%d %I:%M:%S %p"))
    print()
    start = time.time()

    # prepare_features.run()
    smt.run()

    end = time.time()

    print()
    print(datetime.now().strftime("Execution ended at %Y-%m-%d %I:%M:%S %p"))
    print(f"Execution took {round((end - start) * 10**3)} ms")
