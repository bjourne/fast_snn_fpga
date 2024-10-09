#ifndef VAR_VECTOR_H
#define VAR_VECTOR_H

typedef struct {
    void *array;
    size_t capacity;
    size_t used;
    size_t el_size;
} var_vector;

var_vector *var_vec_init(size_t capacity, size_t el_size);
void var_vec_free(var_vector *me);
void var_vec_add(var_vector *me, void *src);
void var_vec_add_sparse(var_vector *me,
                        size_t index, void *src, void *fill);
void var_vec_remove(var_vector *me, void *dst);
void var_vec_remove_at(var_vector *me, size_t i, void *dst);

#endif
