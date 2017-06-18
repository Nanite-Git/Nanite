/*

Copyright (C) 2012-2016 Olaf Till <i7tiol@t-online.de>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; If not, see <http://www.gnu.org/licenses/>.

*/

#ifndef __OCT_PQ_CONVERTERS__

#define __OCT_PQ_CONVERTERS__

#include <octave/oct.h>

#include <stdint.h>

#include <vector>
#include <string>

#include <libpq-fe.h>

#include "wrap_endian.h"

#define OCT_PQ_NUM_CONVERTERS 35

// The following type is used to hold data prepared for transfer from
// the package to the database. There was a discussion, starting here:
// http://lists.gnu.org/archive/html/octave-maintainers/2016-03/msg00183.html
// which implicitely made the use of this type and of the type casts
// in the macros below (OCT_PQ_PUT and the following ones) seem
// questionable.
//
// What I (Olaf) wasn't aware while discussing this is that type
// aliasing always works if one of the types is 'char'. So the type
// casts in the defined macros are actually legitimate, and don't
// require an -fno-strict-aliasing compiler option. So a check for g++
// to be the compiler, which seemed to be the main target of the
// criticism, is actually not necessary. (I'm really sorry for the
// confusion I've possibly caused.)
//
// Caused by the result of the discussion, there was a changeset for
// using 'std::ostringstream' instead of 'std::vector<char>' for data
// preparation. This changeset was backed out again, since using
// 'std::ostringstream' would require copying the data to
// 'std::string' before transmission, which could get quite expensive,
// e.g. with large data in bytea types.
//
// Apart from the check for g++ to be the compiler, the criticism also
// seemed to cover the 'casting tricks' per se in the macros below,
// which were considered to be replacable by a cleaner way. But the
// casting enables the use of a simple assignment instruction, and
// although the compiler or processor will have to perform some
// mangling if the data is not aligned, this instruction is likely to
// enable the compiler to chose the most efficient way for the
// mangling at each system. And it spares the function call, necessary
// if using 'memcpy'.
typedef std::vector<char> oct_pq_dynvec_t;

// append bytes of value 'val' of type 'type' to dynamic char vector 'dv'
#define OCT_PQ_PUT(dv, type, val)                       \
  dv.resize (dv.size () + sizeof (type));               \
  *((type *) &(dv.end ()[-sizeof (type)])) = val;

// increase size of dynamic char vector 'dv' by size of uint32 and
// store the new size in a variable named 'var' of octave_idx_type;
// after further increasing 'dv', OCT_PQ_FILL_UINT32_PLACEHOLDER
// should be used with equal arguments
#define OCT_PQ_SET_UINT32_PLACEHOLDER(dv, var)          \
  octave_idx_type var = dv.size () + sizeof (uint32_t); \
  dv.resize (var);
// to be used after OCT_PQ_SET_UINT32_PLACEHOLDER with equal
// arguments; calculate difference between current size of dynamic
// char vector 'dv' and a previous size stored in a variable named
// 'var', and write this difference, converted to uint32_t in network
// byte order, to the placeholder within 'dv' just before the position
// stored in 'var'
#define OCT_PQ_FILL_UINT32_PLACEHOLDER(dv, var)                         \
  *((uint32_t *) &(dv[var - sizeof (uint32_t)])) = htobe32 (dv.size () - var);

#define OCT_PQ_DECL_GET_INT32(retvar, pointer, type)    \
  type retvar = be32toh (*((type *) pointer));          \
  pointer += 4;

#define OCT_PQ_GET_INT32(retvar, pointer, type)         \
  retvar = be32toh (*((type *) pointer));          \
  pointer += 4;

typedef enum {simple, array, composite} pq_oct_type_t;

typedef struct
{
  dim_vector pd, cur;
  octave_idx_type nd;
}
  oct_mo_count_state;

class octave_pq_connection_rep;

typedef int (*oct_pq_to_octave_fp_t) (const octave_pq_connection_rep &,
                                      const char *, octave_value &, int);

typedef int (*oct_pq_from_octave_fp_t) (const octave_pq_connection_rep &,
                                        const octave_value &,
                                        oct_pq_dynvec_t &);

typedef std::vector<Oid> oct_pq_el_oids_t;

struct oct_pq_conv_t_;

typedef std::vector<struct oct_pq_conv_t_ *> oct_pq_conv_cache_t;

// some objects will be constants, some will be allocated
typedef struct oct_pq_conv_t_
{
  Oid oid; // read from server
  Oid aoid; // array oid // read from server
  oct_pq_el_oids_t el_oids; // element oids, empty for non-composite types
  oct_pq_conv_cache_t conv_cache; // element converter structures for
                                  //composite types
  bool is_composite; // false for constant objects
  bool is_enum; // false for constant objects
  bool is_not_constant; // false for constant objects
  // const char *name; not all constants, use std::string
  std::string name;
  oct_pq_to_octave_fp_t to_octave_str;
  oct_pq_to_octave_fp_t to_octave_bin;
  oct_pq_from_octave_fp_t from_octave_str;
  oct_pq_from_octave_fp_t from_octave_bin;
}
  oct_pq_conv_t;

std::string &pq_basetype_prefix (void);

// a wrapper class for array of pointers to converters which qualifies
// base type names in initialization
class oct_pq_conv_ptrs_t
{
public:

  oct_pq_conv_ptrs_t (int n, oct_pq_conv_t **ptrs) : converters (ptrs)
  {
    for (int i = 0; i < n; i++)
      {
        std::string prefix = pq_basetype_prefix ();

        converters[i]->name = prefix.append (converters[i]->name);
      }
  }

  oct_pq_conv_t *operator[] (int i) { return converters[i]; }

private:

  oct_pq_conv_t **converters;
};

// a wrapper class around oct_pq_conv_t* to provide a default
// constructor which nullifies it, for efficient use of maps, where
// checking for the presence of a key while inserting it can rely on a
// newly generated key mapping to a value of NULL
class oct_pq_conv_wrapper_t
{
public:

  oct_pq_conv_wrapper_t (void) : conv (NULL) {}

  oct_pq_conv_wrapper_t (oct_pq_conv_t *c) : conv (c) {}

  operator oct_pq_conv_t *&(void) { return conv; }

  oct_pq_conv_t *&operator->(void) { return conv; }

  oct_pq_conv_t *get_copy (void) const { return conv; }

private:

  oct_pq_conv_t *conv;
};

typedef std::map<Oid, oct_pq_conv_wrapper_t> oct_pq_conv_map_t;

typedef std::map<const char *, oct_pq_conv_wrapper_t,
  bool (*) (const char *, const char *)> oct_pq_name_conv_map_t;

// helper function for debugging
void print_conv (oct_pq_conv_t *);

extern oct_pq_conv_ptrs_t conv_ptrs;

// these prototypes are needed because pointers to these functions are
// stored in the converter structures of each found enum type
int to_octave_str_text (const octave_pq_connection_rep &conn,
                        const char *c, octave_value &ov, int nb);
int to_octave_bin_text (const octave_pq_connection_rep &conn,
                        const char *c, octave_value &ov, int nb);
int from_octave_str_text (const octave_pq_connection_rep &conn,
                          const octave_value &ov, oct_pq_dynvec_t &val);
int from_octave_bin_text (const octave_pq_connection_rep &conn,
                          const octave_value &ov, oct_pq_dynvec_t &val);

oct_pq_conv_t *pgtype_from_octtype (const octave_pq_connection_rep &conn,
                                    const octave_value &);

oct_pq_conv_t *pgtype_from_spec (const octave_pq_connection_rep &, std::string &,
                                 pq_oct_type_t &);

oct_pq_conv_t *pgtype_from_spec (const octave_pq_connection_rep &, Oid,
                                 pq_oct_type_t &);

oct_pq_conv_t *pgtype_from_spec (const octave_pq_connection_rep &, Oid,
                                 oct_pq_conv_t *&, pq_oct_type_t &);

octave_idx_type count_row_major_order (dim_vector &, oct_mo_count_state &,
                                       bool);

int from_octave_bin_array (const octave_pq_connection_rep &conn,
                           const octave_value &oct_arr, oct_pq_dynvec_t &val,
                           oct_pq_conv_t *);

int from_octave_bin_composite (const octave_pq_connection_rep &conn,
                               const octave_value &oct_comp,
                               oct_pq_dynvec_t &val, oct_pq_conv_t *);

int from_octave_str_array (const octave_pq_connection_rep &conn,
                           const octave_value &oct_arr, oct_pq_dynvec_t &val,
                           octave_value &type);

int from_octave_str_composite (const octave_pq_connection_rep &conn,
                               const octave_value &oct_comp,
                               oct_pq_dynvec_t &val, octave_value &type);

int to_octave_bin_array (const octave_pq_connection_rep &conn,
                         const char *, octave_value &, int, oct_pq_conv_t *);

int to_octave_bin_composite (const octave_pq_connection_rep &conn, const char *,
                             octave_value &, int, oct_pq_conv_t *);

int to_octave_str_array (const octave_pq_connection_rep &conn,
                         const char *, octave_value &, int, oct_pq_conv_t *);

int to_octave_str_composite (const octave_pq_connection_rep &conn, const char *,
                             octave_value &, int, oct_pq_conv_t *);

#endif // __OCT_PQ_CONVERTERS__
