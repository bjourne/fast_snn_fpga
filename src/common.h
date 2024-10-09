#ifndef COMMON_H
#define COMMON_H

// Basic logic
#define MAX(a, b) ((a > b) ? (a) : (b))
#define MIN(a, b) ((a > b) ? (b) : (a))

#define ARRAY_SIZE(a)       (sizeof((a))/sizeof((a)[0]))

#define MACRO_STR(m)            #m
#define MACRO_VAL_STR(m)        MACRO_STR(m)
#define MACRO_KEY_VAL(m)        #m " = " MACRO_VAL_STR(m)

#endif
