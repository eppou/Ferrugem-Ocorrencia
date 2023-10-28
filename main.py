import time
from datetime import datetime

import procedures.prepare_ferrugem_ocorrencia_instances as prepare_instances
import procedures.prepare_ferrugem_ocorrencia_features as prepare_features

if __name__ == "__main__":
    print(datetime.now().strftime("Execution started at %Y-%m-%d %I:%M:%S %p"))
    start = time.time()

    prepare_features.run()

    end = time.time()

    print(datetime.now().strftime("Execution ended at %Y-%m-%d %I:%M:%S %p"))
    print(f"Execution took {round((end - start) * 10**3)} ms")
