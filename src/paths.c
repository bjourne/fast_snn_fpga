// Copyright (C) 2019, 2022-2023 Björn Lindqvist <bjourne@gmail.com>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"
#include "paths.h"

// These functions works almost like
// http://man7.org/linux/man-pages/man3/basename.3.html, except that
// they support backslashes which are used on Windows. On Unix, a
// backslash can be part of a filename so they are technically not
// correct.
// Utility

// Non-allocating functions
const char *
paths_basename(const char *path) {
    char *p1 = strrchr(path, '\\');
    char *p2 = strrchr(path, '/');
    char *p = MAX(p1, p2);
    if (!p) {
        return path;
    }
    return ++p;
}

const char *
paths_ext(const char *fname) {
    char *dot = strrchr(fname, '.');
    if(!dot || dot == fname)
        return "";
    return dot + 1;
}

// Allocating functions
char *
paths_stem(const char *path) {
    const char *path2 = paths_basename(path);
    char *str = strdup(path2);
    char *p = strchr(str, '.');
    if (p) {
        *p = '\0';
    }
    return str;
}

// There are some weird edge cases which I just ignore here.
char *
paths_dirname(char *path) {
    char *p1 = strrchr(path, '\\');
    char *p2 = strrchr(path, '/');
    char *p = MAX(p1, p2);
    if (!p) {
        return strdup("");
    }
    while ((*p == '/' || *p == '\\') && p != path) {
        p--;
    }
    p++;
    int len = (int)(p - path);
    char *buf = (char *)malloc(len + 1);
    strncpy(buf, path, len);
    buf[len] = '\0';
    return buf;
}

char *
paths_normalize(const char *path) {
    size_t n = strlen(path);
    char *ret = malloc(n + 2);
    char *buf = malloc(n);
    size_t ri = 0;
    size_t bi = 0;
    bool is_first = true;
    if (path[0] == '/') {
        ret[0] = '/';
        ri++;
    }
    for (size_t pi = 0; pi <= n; pi++) {
        // Note short-circuiting
        if (pi == n || path[pi] == '/') {
            // Check buffer
            if (bi > 0 && !(bi == 1 && buf[0] == '.')) {
                if (!is_first) {
                    ret[ri] = '/';
                    ri++;
                }
                is_first = false;
                memcpy(&ret[ri], buf, bi);
                ri += bi;
            }
            bi = 0;
        } else {
            buf[bi] = path[pi];
            bi++;
        }
    }
    if (!ri) {
        ret[ri] = '.';
        ri++;
    }
    ret[ri] = '\0';
    free(buf);
    return ret;
}

#include <stdio.h>

char *paths_join(const char *p1, const char *p2) {
    char *x = paths_normalize(p1);
    char *y = paths_normalize(p2);

    size_t xl = strlen(x);
    size_t yl = strlen(y);

    char *z = NULL;
    if (y[0] == '/') {
        z = strdup(y);
    } else {
        char *tmp = malloc(xl + yl + 2);
        memcpy(&tmp[0], x, xl);
        tmp[xl] = '/';
        memcpy(&tmp[xl + 1], y, yl);
        tmp[xl + yl + 1] = '\0';
        z = paths_normalize(tmp);
        free(tmp);
    }
    free(x);
    free(y);
    return z;
}
