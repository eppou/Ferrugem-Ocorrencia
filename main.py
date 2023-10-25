import time

import procedures.prepare_ferrugem_ocorrencia_features as prepare_features

if __name__ == "__main__":
    start = time.time()

    prepare_features.run()

    end = time.time()
    print(f"Execution took {round((end - start) * 10**3)} ms")
