import time

import procedures.prepare_ferrugem_ocorrencia_dataset as p

if __name__ == "__main__":
    start = time.time()

    p.run()

    end = time.time()
    print(f"Execution took {round((end - start) * 10**3)} ms")
