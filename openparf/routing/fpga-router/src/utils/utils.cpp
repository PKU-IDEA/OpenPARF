#include "utils.h"
#include <stdio.h>

double get_memory_peak() {
    struct rusage rusage;
    getrusage(RUSAGE_SELF, &rusage);
    return rusage.ru_maxrss / 1024.0;
}

double get_memory_current() {
    long rss = 0L;
    FILE* fp = NULL;
    if ((fp = fopen("/proc/self/statm", "r")) == NULL) {
        return 0.0; /* Can't open? */
    }
    if (fscanf(fp, "%*s%ld", &rss) != 1) {
        fclose(fp);
        return 0.0; /* Can't read? */
    }
    fclose(fp);
    return rss * sysconf(_SC_PAGESIZE) / 1048576.0;
}