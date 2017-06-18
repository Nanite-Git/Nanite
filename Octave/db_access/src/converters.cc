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

#include <octave/oct.h>
#include <octave/ov-struct.h>
#include <octave/ov-float.h>
#include <octave/ov-uint8.h>
#include <octave/Cell.h>

#include <libpq-fe.h>

#ifdef __WIN32__
#include <winsock2.h>
#else
#include <sys/socket.h> // for AF_INET, needed in network address types
#endif

#include "converters.h"
#include "pq_connection.h"
#include "error-helpers.h"

// remember to adjust OCT_PQ_NUM_CONVERTERS in converters.h

#define PGSQL_AF_INET6 (AF_INET + 1) // defined so in postgresql, no
                                     // public header file available

// helper function for debugging
void print_conv (oct_pq_conv_t *c)
{
  printf ("oid: %u, aoid: %u, c: %i, e: %i, nc: %i, n: %s, to_s: %i, to_b: %i, fr_s: %i, fr_b: %i,",
          c->oid, c->aoid, c->is_composite, c->is_enum, c->is_not_constant, c->name.c_str (), c->to_octave_str ? 1 : 0, c->to_octave_bin ? 1 : 0, c->from_octave_str ? 1 : 0, c->from_octave_bin ? 1 : 0);

  printf (", el_oids:");
  for (size_t i = 0; i < c->el_oids.size (); i++)
    printf (" %u", c->el_oids[i]);

  printf ("\n");
}

/* type bool */

int to_octave_str_bool (const octave_pq_connection_rep &conn,
                        const char *c, octave_value &ov, int nb)
{
  bool tp = (*c == 't' ? true : false);

  ov = octave_value (tp);

  return 0;
}

int to_octave_bin_bool (const octave_pq_connection_rep &conn,
                        const char *c, octave_value &ov, int nb)
{
  ov = octave_value (bool (*c));

  return 0;
}

int from_octave_str_bool (const octave_pq_connection_rep &conn,
                          const octave_value &ov, oct_pq_dynvec_t &val)
{
  bool b, err;
  SET_ERR (b = ov.bool_value (), err);

  if (err)
    {
      c_verror ("can not convert octave_value to bool value");
      return 1;
    }

  val.push_back (b ? '1' : '0');
  val.push_back ('\0');

  return 0;
}

int from_octave_bin_bool (const octave_pq_connection_rep &conn,
                          const octave_value &ov, oct_pq_dynvec_t &val)
{
  bool b, err;
  SET_ERR (b = ov.bool_value (), err);

  if (err)
    {
      c_verror ("can not convert octave_value to bool value");
      return 1;
    }

  val.push_back (char (b));

  return 0;
}

oct_pq_conv_t conv_bool = {0, // 16
                           0, // 1000
                           oct_pq_el_oids_t (),
                           oct_pq_conv_cache_t (),
                           false,
                           false,
                           false,
                           "bool",
                           &to_octave_str_bool,
                           &to_octave_bin_bool,
                           &from_octave_str_bool,
                           &from_octave_bin_bool};

/* end type bool */

/* type oid */

int to_octave_str_oid (const octave_pq_connection_rep &conn,
                       const char *c, octave_value &ov, int nb)
{
  return 1;
}

int to_octave_bin_oid (const octave_pq_connection_rep &conn,
                       const char *c, octave_value &ov, int nb)
{
  ov = octave_value (octave_uint32 (be32toh (*((uint32_t *) c))));

  return 0;
}

int from_octave_str_oid (const octave_pq_connection_rep &conn,
                         const octave_value &ov, oct_pq_dynvec_t &val)
{
  return 1;
}

int from_octave_bin_oid (const octave_pq_connection_rep &conn,
                         const octave_value &ov, oct_pq_dynvec_t &val)
{
  uint32_t oid;
  bool err;
  SET_ERR (oid = ov.uint_value (), err);

  if (err)
    {
      c_verror ("can not convert octave_value to oid value");
      return 1;
    }

  OCT_PQ_PUT(val, uint32_t, htobe32 (oid))

  return 0;
}

oct_pq_conv_t conv_oid = {0, // 26
                          0,
                          oct_pq_el_oids_t (),
                          oct_pq_conv_cache_t (),
                          false,
                          false,
                          false,
                          "oid",
                          &to_octave_str_oid,
                          &to_octave_bin_oid,
                          &from_octave_str_oid,
                          &from_octave_bin_oid};

/* end type oid */

/* type float8 */

int to_octave_str_float8 (const octave_pq_connection_rep &conn,
                          const char *c, octave_value &ov, int nb)
{
  // not implemented

  return 1;
}

int to_octave_bin_float8 (const octave_pq_connection_rep &conn,
                          const char *c, octave_value &ov, int nb)
{
  union
  {
    double d;
    int64_t i;
  }
  swap;

  swap.i = be64toh (*((int64_t *) c));

  ov = octave_value (swap.d);

  return 0;
}

int from_octave_str_float8 (const octave_pq_connection_rep &conn,
                            const octave_value &ov, oct_pq_dynvec_t &val)
{
  // not implemented

  return 1;
}

int from_octave_bin_float8 (const octave_pq_connection_rep &conn,
                            const octave_value &ov, oct_pq_dynvec_t &val)
{
  union
  {
    double d;
    int64_t i;
  }
  swap;

  bool err;
  SET_ERR (swap.d = ov.double_value (), err);

  if (err)
    {
      c_verror ("can not convert octave_value to float8 value");
      return 1;
    }

  OCT_PQ_PUT(val, int64_t, htobe64 (swap.i))

  return 0;
}

oct_pq_conv_t conv_float8 = {0, // 701
                             0, // 1022
                             oct_pq_el_oids_t (),
                             oct_pq_conv_cache_t (),
                             false,
                             false,
                             false,
                             "float8",
                             &to_octave_str_float8,
                             &to_octave_bin_float8,
                             &from_octave_str_float8,
                             &from_octave_bin_float8};

/* end type float8 */

/* type float4 */

int to_octave_str_float4 (const octave_pq_connection_rep &conn,
                          const char *c, octave_value &ov, int nb)
{
  // not implemented

  return 1;
}

int to_octave_bin_float4 (const octave_pq_connection_rep &conn,
                          const char *c, octave_value &ov, int nb)
{
  union
  {
    float f;
    int32_t i;
  }
  swap;

  swap.i = be32toh (*((int32_t *) c));

  ov = octave_value (swap.f);

  return 0;
}

int from_octave_str_float4 (const octave_pq_connection_rep &conn,
                            const octave_value &ov, oct_pq_dynvec_t &val)
{
  // not implemented

  return 1;
}

int from_octave_bin_float4 (const octave_pq_connection_rep &conn,
                            const octave_value &ov, oct_pq_dynvec_t &val)
{
  union
  {
    float f;
    int32_t i;
  }
  swap;

  bool err;
  SET_ERR (swap.f = ov.float_value (), err);

  if (err)
    {
      c_verror ("can not convert octave_value to float4 value");
      return 1;
    }

  OCT_PQ_PUT(val, int32_t, htobe32 (swap.i))

  return 0;
}

oct_pq_conv_t conv_float4 = {0,
                             0,
                             oct_pq_el_oids_t (),
                             oct_pq_conv_cache_t (),
                             false,
                             false,
                             false,
                             "float4",
                             &to_octave_str_float4,
                             &to_octave_bin_float4,
                             &from_octave_str_float4,
                             &from_octave_bin_float4};

/* end type float4 */

/* type bytea */

int to_octave_str_bytea (const octave_pq_connection_rep &conn,
                         const char *c, octave_value &ov, int nb)
{
  return 1;
}

int to_octave_bin_bytea (const octave_pq_connection_rep &conn,
                         const char *c, octave_value &ov, int nb)
{
  uint8NDArray m (dim_vector (nb, 1));

  memcpy (m.fortran_vec (), c, nb);

  ov = octave_value (m);

  return 0;
}

int from_octave_str_bytea (const octave_pq_connection_rep &conn,
                           const octave_value &ov, oct_pq_dynvec_t &val)
{
  return 1;
}

int from_octave_bin_bytea (const octave_pq_connection_rep &conn,
                           const octave_value &ov, oct_pq_dynvec_t &val)
{
  uint8NDArray b;
  bool err;
  SET_ERR (b = ov.uint8_array_value (), err);

  if (err)
    {
      c_verror ("can not convert octave_value to bytea representation");
      return 1;
    }

  octave_idx_type nl = b.numel ();

  octave_idx_type csize = val.size ();

  val.resize (csize + nl);

  memcpy (&(val[csize]), b.fortran_vec (), nl);

  return 0;
}

oct_pq_conv_t conv_bytea = {0,
                            0,
                            oct_pq_el_oids_t (),
                            oct_pq_conv_cache_t (),
                            false,
                            false,
                            false,
                            "bytea",
                            &to_octave_str_bytea,
                            &to_octave_bin_bytea,
                            &from_octave_str_bytea,
                            &from_octave_bin_bytea};

/* end type bytea */

/* type text */

int to_octave_str_text (const octave_pq_connection_rep &conn,
                        const char *c, octave_value &ov, int nb)
{
  return 1;
}

int to_octave_bin_text (const octave_pq_connection_rep &conn,
                        const char *c, octave_value &ov, int nb)
{
  std::string s (c, nb);

  ov = octave_value (s);

  return 0;
}

int from_octave_str_text (const octave_pq_connection_rep &conn,
                          const octave_value &ov, oct_pq_dynvec_t &val)
{
  return 1;
}

int from_octave_bin_text (const octave_pq_connection_rep &conn,
                          const octave_value &ov, oct_pq_dynvec_t &val)
{
  std::string s;
  bool err;
  SET_ERR (s = ov.string_value (), err);

  if (err)
    {
      c_verror ("can not convert octave_value to string");
      return 1;
    }

  octave_idx_type l = s.size ();

  octave_idx_type csize = val.size ();

  val.resize (csize + l);

  memcpy (&(val[csize]), s.data (), l);

  return 0;
}

oct_pq_conv_t conv_text = {0,
                           0,
                           oct_pq_el_oids_t (),
                           oct_pq_conv_cache_t (),
                           false,
                           false,
                           false,
                           "text",
                           &to_octave_str_text,
                           &to_octave_bin_text,
                           &from_octave_str_text,
                           &from_octave_bin_text};

/* end type text */

/* type varchar */

oct_pq_conv_t conv_varchar = {0,
                              0,
                              oct_pq_el_oids_t (),
                              oct_pq_conv_cache_t (),
                              false,
                              false,
                              false,
                              "varchar",
                              &to_octave_str_text,
                              &to_octave_bin_text,
                              &from_octave_str_text,
                              &from_octave_bin_text};

/* end type varchar */

/* type bpchar */

oct_pq_conv_t conv_bpchar = {0,
                             0,
                             oct_pq_el_oids_t (),
                             oct_pq_conv_cache_t (),
                             false,
                             false,
                             false,
                             "bpchar",
                             &to_octave_str_text,
                             &to_octave_bin_text,
                             &from_octave_str_text,
                             &from_octave_bin_text};

/* end type bpchar */

/* type name */

oct_pq_conv_t conv_name = {0,
                           0,
                           oct_pq_el_oids_t (),
                           oct_pq_conv_cache_t (),
                           false,
                           false,
                           false,
                           "name",
                           &to_octave_str_text,
                           &to_octave_bin_text,
                           &from_octave_str_text,
                           &from_octave_bin_text};

/* end type name */

/* type int2 */

int to_octave_str_int2 (const octave_pq_connection_rep &conn,
                        const char *c, octave_value &ov, int nb)
{
  return 1;
}

int to_octave_bin_int2 (const octave_pq_connection_rep &conn,
                        const char *c, octave_value &ov, int nb)
{
  ov = octave_value (octave_int16 (int16_t (be16toh (*((int16_t *) c)))));

  return 0;
}

int from_octave_str_int2 (const octave_pq_connection_rep &conn,
                          const octave_value &ov, oct_pq_dynvec_t &val)
{
  return 1;
}

int from_octave_bin_int2 (const octave_pq_connection_rep &conn,
                          const octave_value &ov, oct_pq_dynvec_t &val)
{
  int16_t i2;
  bool err;
  SET_ERR (i2 = ov.int_value (), err);

  if (err)
    {
      c_verror ("can not convert octave_value to int2 value");
      return 1;
    }

  OCT_PQ_PUT(val, int16_t, htobe16 (i2))

  return 0;
}

oct_pq_conv_t conv_int2 = {0, // 26
                           0,
                           oct_pq_el_oids_t (),
                           oct_pq_conv_cache_t (),
                           false,
                           false,
                           false,
                           "int2",
                           &to_octave_str_int2,
                           &to_octave_bin_int2,
                           &from_octave_str_int2,
                           &from_octave_bin_int2};


/* end type int2 */

/* type int4 */

int to_octave_str_int4 (const octave_pq_connection_rep &conn,
                        const char *c, octave_value &ov, int nb)
{
  return 1;
}

int to_octave_bin_int4 (const octave_pq_connection_rep &conn,
                        const char *c, octave_value &ov, int nb)
{
  ov = octave_value (octave_int32 (int32_t (be32toh (*((int32_t *) c)))));

  return 0;
}

int from_octave_str_int4 (const octave_pq_connection_rep &conn,
                          const octave_value &ov, oct_pq_dynvec_t &val)
{
  return 1;
}

int from_octave_bin_int4 (const octave_pq_connection_rep &conn,
                          const octave_value &ov, oct_pq_dynvec_t &val)
{
  int32_t i4;
  bool err;
  SET_ERR (i4 = ov.int_value (), err);

  if (err)
    {
      c_verror ("can not convert octave_value to int4 value");
      return 1;
    }

  OCT_PQ_PUT(val, int32_t, htobe32 (i4))

  return 0;
}

oct_pq_conv_t conv_int4 = {0,
                           0,
                           oct_pq_el_oids_t (),
                           oct_pq_conv_cache_t (),
                           false,
                           false,
                           false,
                           "int4",
                           &to_octave_str_int4,
                           &to_octave_bin_int4,
                           &from_octave_str_int4,
                           &from_octave_bin_int4};


/* end type int4 */

/* type int8 */

int to_octave_str_int8 (const octave_pq_connection_rep &conn,
                        const char *c, octave_value &ov, int nb)
{
  return 1;
}

int to_octave_bin_int8 (const octave_pq_connection_rep &conn,
                        const char *c, octave_value &ov, int nb)
{
  ov = octave_value (octave_int64 (int64_t (be64toh (*((int64_t *) c)))));

  return 0;
}

int from_octave_str_int8 (const octave_pq_connection_rep &conn,
                          const octave_value &ov, oct_pq_dynvec_t &val)
{
  return 1;
}

int from_octave_bin_int8 (const octave_pq_connection_rep &conn,
                          const octave_value &ov, oct_pq_dynvec_t &val)
{
  int64_t i8;
  bool err;
  SET_ERR (i8 = ov.int64_scalar_value (), err);

  if (err)
    {
      c_verror ("can not convert octave_value to int8 value");
      return 1;
    }

  OCT_PQ_PUT(val, int64_t, htobe64 (i8))

  return 0;
}

oct_pq_conv_t conv_int8 = {0,
                           0,
                           oct_pq_el_oids_t (),
                           oct_pq_conv_cache_t (),
                           false,
                           false,
                           false,
                           "int8",
                           &to_octave_str_int8,
                           &to_octave_bin_int8,
                           &from_octave_str_int8,
                           &from_octave_bin_int8};


/* end type int8 */

/* type money */

oct_pq_conv_t conv_money = {0,
                            0,
                            oct_pq_el_oids_t (),
                            oct_pq_conv_cache_t (),
                            false,
                            false,
                            false,
                            "money",
                            &to_octave_str_int8,
                            &to_octave_bin_int8,
                            &from_octave_str_int8,
                            &from_octave_bin_int8};

/* end type money */

// helpers for time types

static inline octave_value time_8byte_to_octave (const char *c,
                                                 const bool &int_dt)
{
  if (int_dt)
    {
      return octave_value (octave_int64 (int64_t (be64toh (*((int64_t *) c)))));
    }
  else
    {
      union
      {
        double d;
        int64_t i;
      }
      swap;

      swap.i = be64toh (*((int64_t *) c));

      return octave_value (swap.d);
    }
}

static inline int time_8byte_from_octave (const octave_value &ov,
                                          oct_pq_dynvec_t &val,
                                          const bool &int_dt)
{
  if (int_dt)
    {
      // don't convert automatically because of possible overflow
      if (ov.is_float_type ())
        {
          c_verror ("floating point octave_value provided for 8-byte time value, but postgresql is configured for int64");
          return 1;
        }

      int64_t i8;
      bool err;
      SET_ERR (i8 = ov.int64_scalar_value (), err);

      if (err)
        {
          c_verror ("can not convert octave_value to int64 time value");
          return 1;
        }

      OCT_PQ_PUT(val, int64_t, htobe64 (i8))

      return 0;
    }
  else
    {
      // don't convert automatically because of possible loss of accuracy
      if (ov.is_integer_type ())
        {
          c_verror ("integer type octave_value provided for 8-byte time value, but postgresql is configured for double");
          return 1;
        }

      union
      {
        double d;
        int64_t i;
      }
      swap;

      bool err;
      SET_ERR (swap.d = ov.double_value (), err);

      if (err)
        {
          c_verror ("can not convert octave_value to double time value");
          return 1;
        }

      OCT_PQ_PUT(val, int64_t, htobe64 (swap.i))

      return 0;
    }
}

// end helpers for time types

/* type timestamp */

int to_octave_str_timestamp (const octave_pq_connection_rep &conn,
                             const char *c, octave_value &ov, int nb)
{
  return 1;
}

int to_octave_bin_timestamp (const octave_pq_connection_rep &conn,
                             const char *c, octave_value &ov, int nb)
{
  ov = time_8byte_to_octave (c, conn.get_integer_datetimes ());

  return 0;
}

int from_octave_str_timestamp (const octave_pq_connection_rep &conn,
                               const octave_value &ov, oct_pq_dynvec_t &val)
{
  return 1;
}

int from_octave_bin_timestamp (const octave_pq_connection_rep &conn,
                               const octave_value &ov, oct_pq_dynvec_t &val)
{
  return (time_8byte_from_octave (ov, val, conn.get_integer_datetimes ()));
}

oct_pq_conv_t conv_timestamp = {0,
                                0,
                                oct_pq_el_oids_t (),
                                oct_pq_conv_cache_t (),
                                false,
                                false,
                                false,
                                "timestamp",
                                &to_octave_str_timestamp,
                                &to_octave_bin_timestamp,
                                &from_octave_str_timestamp,
                                &from_octave_bin_timestamp};

/* end type timestamp */

/* type timestamptz */

oct_pq_conv_t conv_timestamptz = {0,
                                  0,
                                  oct_pq_el_oids_t (),
                                  oct_pq_conv_cache_t (),
                                  false,
                                  false,
                                  false,
                                  "timestamptz",
                                  &to_octave_str_timestamp,
                                  &to_octave_bin_timestamp,
                                  &from_octave_str_timestamp,
                                  &from_octave_bin_timestamp};

/* end type timestamptz */

/* type interval */

int to_octave_str_interval (const octave_pq_connection_rep &conn,
                            const char *c, octave_value &ov, int nb)
{
  return 1;
}

int to_octave_bin_interval (const octave_pq_connection_rep &conn,
                            const char *c, octave_value &ov, int nb)
{
  Cell tp (dim_vector (3, 1));

  tp(0) = time_8byte_to_octave (c, conn.get_integer_datetimes ());

  c += 8;

  tp(1) = octave_value (octave_int32 (int32_t (be32toh (*((int32_t *) c)))));

  c += 4;

  tp(2) = octave_value (octave_int32 (int32_t (be32toh (*((int32_t *) c)))));

  ov = octave_value (tp);

  return 0;
}

int from_octave_str_interval (const octave_pq_connection_rep &conn,
                              const octave_value &ov, oct_pq_dynvec_t &val)
{
  return 1;
}

int from_octave_bin_interval (const octave_pq_connection_rep &conn,
                              const octave_value &ov, oct_pq_dynvec_t &val)
{
  Cell iv;
  bool err;
  SET_ERR (iv = ov.cell_value (), err);
  if (err || iv.numel () != 3)
    {
      c_verror ("interval: can not convert octave_value to cell with 3 elements");
      return 1;
    }

  if (time_8byte_from_octave (iv(0), val, conn.get_integer_datetimes ()))
    return 1;

  for (int id = 1; id < 3; id++)
    {
      int32_t i4;
      bool err;
      SET_ERR (i4 = iv(id).int_value (), err);

      if (err)
        {
          c_verror ("interval: can not convert octave_value to int4 value");
          return 1;
        }

      OCT_PQ_PUT(val, int32_t, htobe32 (i4))
    }

  return 0;
}

oct_pq_conv_t conv_interval = {0,
                               0,
                               oct_pq_el_oids_t (),
                               oct_pq_conv_cache_t (),
                               false,
                               false,
                               false,
                               "interval",
                               &to_octave_str_interval,
                               &to_octave_bin_interval,
                               &from_octave_str_interval,
                               &from_octave_bin_interval};

/* end type interval */

/* type time */

int to_octave_str_time (const octave_pq_connection_rep &conn,
                        const char *c, octave_value &ov, int nb)
{
  return 1;
}

int to_octave_bin_time (const octave_pq_connection_rep &conn,
                        const char *c, octave_value &ov, int nb)
{
  ov = time_8byte_to_octave (c, conn.get_integer_datetimes ());

  return 0;
}

int from_octave_str_time (const octave_pq_connection_rep &conn,
                          const octave_value &ov, oct_pq_dynvec_t &val)
{
  return 1;
}

int from_octave_bin_time (const octave_pq_connection_rep &conn,
                          const octave_value &ov, oct_pq_dynvec_t &val)
{
  return (time_8byte_from_octave (ov, val, conn.get_integer_datetimes ()));
}

oct_pq_conv_t conv_time = {0,
                           0,
                           oct_pq_el_oids_t (),
                           oct_pq_conv_cache_t (),
                           false,
                           false,
                           false,
                           "time",
                           &to_octave_str_time,
                           &to_octave_bin_time,
                           &from_octave_str_time,
                           &from_octave_bin_time};

/* end type time */

/* type timetz */

int to_octave_str_timetz (const octave_pq_connection_rep &conn,
                          const char *c, octave_value &ov, int nb)
{
  return 1;
}

int to_octave_bin_timetz (const octave_pq_connection_rep &conn,
                          const char *c, octave_value &ov, int nb)
{
  Cell tp (dim_vector (2, 1));

  tp(0) = time_8byte_to_octave (c, conn.get_integer_datetimes ());

  c += 8;

  tp(1) = octave_value (octave_int32 (int32_t (be32toh (*((int32_t *) c)))));

  ov = octave_value (tp);

  return 0;
}

int from_octave_str_timetz (const octave_pq_connection_rep &conn,
                            const octave_value &ov, oct_pq_dynvec_t &val)
{
  return 1;
}

int from_octave_bin_timetz (const octave_pq_connection_rep &conn,
                            const octave_value &ov, oct_pq_dynvec_t &val)
{
  Cell iv;
  bool err;
  SET_ERR (iv = ov.cell_value (), err);
  if (err || iv.numel () != 2)
    {
      c_verror ("timetz: can not convert octave_value to cell with 2 elements");
      return 1;
    }

  if (time_8byte_from_octave (iv(0), val, conn.get_integer_datetimes ()))
    return 1;

  int32_t i4;
  SET_ERR (i4 = iv(1).int_value (), err);

  if (err)
    {
      c_verror ("timetz: can not convert octave_value to int4 value");
      return 1;
    }

  OCT_PQ_PUT(val, int32_t, htobe32 (i4))

  return 0;
}

oct_pq_conv_t conv_timetz = {0,
                             0,
                             oct_pq_el_oids_t (),
                             oct_pq_conv_cache_t (),
                             false,
                             false,
                             false,
                             "timetz",
                             &to_octave_str_timetz,
                             &to_octave_bin_timetz,
                             &from_octave_str_timetz,
                             &from_octave_bin_timetz};

/* end type timetz */

/* type date */

int to_octave_str_date (const octave_pq_connection_rep &conn,
                        const char *c, octave_value &ov, int nb)
{
  return 1;
}

int to_octave_bin_date (const octave_pq_connection_rep &conn,
                        const char *c, octave_value &ov, int nb)
{
  ov = octave_value (octave_int32 (int32_t (be32toh (*((int32_t *) c)))));

  return 0;
}

int from_octave_str_date (const octave_pq_connection_rep &conn,
                          const octave_value &ov, oct_pq_dynvec_t &val)
{
  return 1;
}

int from_octave_bin_date (const octave_pq_connection_rep &conn,
                          const octave_value &ov, oct_pq_dynvec_t &val)
{
  int32_t i4;
  bool err;
  SET_ERR (i4 = ov.int_value (), err);

  if (err)
    {
      c_verror ("date: can not convert octave_value to int4 value");
      return 1;
    }

  OCT_PQ_PUT(val, int32_t, htobe32 (i4))

  return 0;
}

oct_pq_conv_t conv_date = {0,
                           0,
                           oct_pq_el_oids_t (),
                           oct_pq_conv_cache_t (),
                           false,
                           false,
                           false,
                           "date",
                           &to_octave_str_date,
                           &to_octave_bin_date,
                           &from_octave_str_date,
                           &from_octave_bin_date};

/* end type date */

/* type point */

int to_octave_str_point (const octave_pq_connection_rep &conn,
                         const char *c, octave_value &ov, int nb)
{
  return 1;
}

int to_octave_bin_point (const octave_pq_connection_rep &conn,
                         const char *c, octave_value &ov, int nb)
{
  ColumnVector m (2);

  union
  {
    double d;
    int64_t i;
  }
  swap;

  for (int id = 0; id < 2; id++, c += 8)
    {
      swap.i = be64toh (*((int64_t *) c));

      m(id) = swap.d;
    }

  ov = octave_value (m);

  return 0;
}

int from_octave_str_point (const octave_pq_connection_rep &conn,
                           const octave_value &ov, oct_pq_dynvec_t &val)
{
  return 1;
}

int from_octave_bin_point (const octave_pq_connection_rep &conn,
                           const octave_value &ov, oct_pq_dynvec_t &val)
{
  NDArray m;
  bool err;
  SET_ERR (m = ov.array_value (), err);

  if (err || m.numel () != 2)
    {
      c_verror ("can not convert octave_value to point representation");
      return 1;
    }

  union
  {
    double d;
    int64_t i;
  }
  swap;

  for (int id = 0; id < 2; id++)
    {
      swap.d = m(id);

      OCT_PQ_PUT(val, int64_t, htobe64 (swap.i))
    }

  return 0;
}

oct_pq_conv_t conv_point = {0,
                            0,
                            oct_pq_el_oids_t (),
                            oct_pq_conv_cache_t (),
                            false,
                            false,
                            false,
                            "point",
                            &to_octave_str_point,
                            &to_octave_bin_point,
                            &from_octave_str_point,
                            &from_octave_bin_point};

/* end type point */

/* type lseg */

int to_octave_str_lseg (const octave_pq_connection_rep &conn,
                        const char *c, octave_value &ov, int nb)
{
  return 1;
}

int to_octave_bin_lseg (const octave_pq_connection_rep &conn,
                        const char *c, octave_value &ov, int nb)
{
  Matrix m (2, 2);

  union
  {
    double d;
    int64_t i;
  }
  swap;

  for (int id = 0; id < 4; id++, c += 8)
    {
      swap.i = be64toh (*((int64_t *) c));

      m(id) = swap.d;
    }

  ov = octave_value (m);

  return 0;
}

int from_octave_str_lseg (const octave_pq_connection_rep &conn,
                          const octave_value &ov, oct_pq_dynvec_t &val)
{
  return 1;
}

int from_octave_bin_lseg (const octave_pq_connection_rep &conn,
                          const octave_value &ov, oct_pq_dynvec_t &val)
{
  NDArray m;
  bool err;
  SET_ERR (m = ov.array_value (), err);

  if (err || m.numel () != 4)
    {
      c_verror ("can not convert octave_value to 4 doubles");
      return 1;
    }

  union
  {
    double d;
    int64_t i;
  }
  swap;

  for (int id = 0; id < 4; id++)
    {
      swap.d = m(id);

      OCT_PQ_PUT(val, int64_t, htobe64 (swap.i))
    }

  return 0;
}

oct_pq_conv_t conv_lseg = {0,
                           0,
                           oct_pq_el_oids_t (),
                           oct_pq_conv_cache_t (),
                           false,
                           false,
                           false,
                           "lseg",
                           &to_octave_str_lseg,
                           &to_octave_bin_lseg,
                           &from_octave_str_lseg,
                           &from_octave_bin_lseg};

/* end type lseg */

/* type line */

oct_pq_conv_t conv_line = {0,
                           0,
                           oct_pq_el_oids_t (),
                           oct_pq_conv_cache_t (),
                           false,
                           false,
                           false,
                           "line",
                           &to_octave_str_lseg,
                           &to_octave_bin_lseg,
                           &from_octave_str_lseg,
                           &from_octave_bin_lseg};

/* end type line */

/* type box */

oct_pq_conv_t conv_box = {0,
                          0,
                          oct_pq_el_oids_t (),
                          oct_pq_conv_cache_t (),
                          false,
                          false,
                          false,
                          "box",
                          &to_octave_str_lseg,
                          &to_octave_bin_lseg,
                          &from_octave_str_lseg,
                          &from_octave_bin_lseg};

/* end type box */

/* type circle */

int to_octave_str_circle (const octave_pq_connection_rep &conn,
                          const char *c, octave_value &ov, int nb)
{
  return 1;
}

int to_octave_bin_circle (const octave_pq_connection_rep &conn,
                          const char *c, octave_value &ov, int nb)
{
  ColumnVector m (3);

  union
  {
    double d;
    int64_t i;
  }
  swap;

  for (int id = 0; id < 3; id++, c += 8)
    {
      swap.i = be64toh (*((int64_t *) c));

      m(id) = swap.d;
    }

  ov = octave_value (m);

  return 0;
}

int from_octave_str_circle (const octave_pq_connection_rep &conn,
                            const octave_value &ov, oct_pq_dynvec_t &val)
{
  return 1;
}

int from_octave_bin_circle (const octave_pq_connection_rep &conn,
                            const octave_value &ov, oct_pq_dynvec_t &val)
{
  NDArray m;
  bool err;
  SET_ERR (m = ov.array_value (), err);

  if (err || m.numel () != 3)
    {
      c_verror ("can not convert octave_value to circle representation");
      return 1;
    }

  union
  {
    double d;
    int64_t i;
  }
  swap;

  for (int id = 0; id < 3; id++)
    {
      swap.d = m(id);

      OCT_PQ_PUT(val, int64_t, htobe64 (swap.i))
    }

  return 0;
}

oct_pq_conv_t conv_circle = {0,
                             0,
                             oct_pq_el_oids_t (),
                             oct_pq_conv_cache_t (),
                             false,
                             false,
                             false,
                             "circle",
                             &to_octave_str_circle,
                             &to_octave_bin_circle,
                             &from_octave_str_circle,
                             &from_octave_bin_circle};

/* end type circle */

/* type polygon */

int to_octave_str_polygon (const octave_pq_connection_rep &conn,
                           const char *c, octave_value &ov, int nb)
{
  return 1;
}

int to_octave_bin_polygon (const octave_pq_connection_rep &conn,
                           const char *c, octave_value &ov, int nb)
{
  int32_t np = int32_t (be32toh (*((int32_t *) c)));

  c += 4;

  Matrix m (2, np);

  union
  {
    double d;
    int64_t i;
  }
  swap;

  for (int id = 0; id < np * 2; id++, c += 8)
    {
      swap.i = be64toh (*((int64_t *) c));

      m(id) = swap.d;
    }

  ov = octave_value (m);

  return 0;
}

int from_octave_str_polygon (const octave_pq_connection_rep &conn,
                             const octave_value &ov, oct_pq_dynvec_t &val)
{
  return 1;
}

int from_octave_bin_polygon (const octave_pq_connection_rep &conn,
                             const octave_value &ov, oct_pq_dynvec_t &val)
{
  octave_idx_type nel;

  NDArray m;
  bool err;
  SET_ERR (m = ov.array_value (), err);

  if (err || (nel = m.numel ()) % 2)
    {
      c_verror ("can not convert octave_value to polygon representation");
      return 1;
    }

  union
  {
    double d;
    int64_t i;
  }
  swap;

  int32_t np = nel / 2;

  OCT_PQ_PUT(val, int32_t, htobe32 (np))

  for (int id = 0; id < nel; id++)
    {
      swap.d = m(id);

      OCT_PQ_PUT(val, int64_t, htobe64 (swap.i))
    }

  return 0;
}

oct_pq_conv_t conv_polygon = {0,
                              0,
                              oct_pq_el_oids_t (),
                              oct_pq_conv_cache_t (),
                              false,
                              false,
                              false,
                              "polygon",
                              &to_octave_str_polygon,
                              &to_octave_bin_polygon,
                              &from_octave_str_polygon,
                              &from_octave_bin_polygon};

/* end type polygon */

/* type path */

int to_octave_str_path (const octave_pq_connection_rep &conn,
                        const char *c, octave_value &ov, int nb)
{
  return 1;
}

int to_octave_bin_path (const octave_pq_connection_rep &conn,
                        const char *c, octave_value &ov, int nb)
{
  bool closed = bool (*(c++));

  int32_t np = int32_t (be32toh (*((int32_t *) c)));

  c += 4;

  Matrix m (2, np);

  union
  {
    double d;
    int64_t i;
  }
  swap;

  for (int id = 0; id < np * 2; id++, c += 8)
    {
      swap.i = be64toh (*((int64_t *) c));

      m(id) = swap.d;
    }

  octave_scalar_map tp;
  tp.assign ("closed", octave_value (closed));
  tp.assign ("path", octave_value (m));

  ov = octave_value (tp);

  return 0;
}

int from_octave_str_path (const octave_pq_connection_rep &conn,
                          const octave_value &ov, oct_pq_dynvec_t &val)
{
  return 1;
}

int from_octave_bin_path (const octave_pq_connection_rep &conn,
                          const octave_value &ov, oct_pq_dynvec_t &val)
{
  octave_scalar_map tp;
  bool err;
  SET_ERR (tp = ov.scalar_map_value (), err);
  if (err || ! tp.isfield ("closed") || ! tp.isfield ("path"))
    {
      c_verror ("can not convert octave_value to path representation");
      return 1;
    }

  octave_idx_type nel;

  char closed;
  SET_ERR (closed = char (tp.contents ("closed").bool_value ()), err);

  NDArray m;
  if (! err)
    {
      SET_ERR (m = tp.contents ("path").array_value (), err);
    }

  if (err || (nel = m.numel ()) % 2)
    {
      c_verror ("can not convert octave_value to path representation");
      return 1;
    }

  union
  {
    double d;
    int64_t i;
  }
  swap;

  int32_t np = nel / 2;

  val.push_back (closed);

  OCT_PQ_PUT(val, int32_t, htobe32 (np))

  for (int id = 0; id < nel; id++)
    {
      swap.d = m(id);

      OCT_PQ_PUT(val, int64_t, htobe64 (swap.i))
    }

  return 0;
}

oct_pq_conv_t conv_path = {0,
                           0,
                           oct_pq_el_oids_t (),
                           oct_pq_conv_cache_t (),
                           false,
                           false,
                           false,
                           "path",
                           &to_octave_str_path,
                           &to_octave_bin_path,
                           &from_octave_str_path,
                           &from_octave_bin_path};

/* end type path */

/* type unknown */

// These are pseudo-converters for postgresql type 'unknown'. They do
// nothing except signalling an error. The rationale is that the only
// values of type 'unknown' may be NULL, in which case the converter
// will not be called, but because a converter exists there won't be a
// "no converter found" error thrown.

int to_octave_str_unknown (const octave_pq_connection_rep &conn,
                           const char *c, octave_value &ov, int nb)
{
  c_verror ("can not convert postgresql type 'unknown'");

  return 1;
}

int to_octave_bin_unknown (const octave_pq_connection_rep &conn,
                           const char *c, octave_value &ov, int nb)
{
  c_verror ("can not convert postgresql type 'unknown'");

  return 1;
}

int from_octave_str_unknown (const octave_pq_connection_rep &conn,
                             const octave_value &ov, oct_pq_dynvec_t &val)
{
  c_verror ("can not convert postgresql type 'unknown'");

  return 1;
}

int from_octave_bin_unknown (const octave_pq_connection_rep &conn,
                             const octave_value &ov, oct_pq_dynvec_t &val)
{
  c_verror ("can not convert postgresql type 'unknown'");

  return 1;
}

oct_pq_conv_t conv_unknown = {0,
                              0,
                              oct_pq_el_oids_t (),
                              oct_pq_conv_cache_t (),
                              false,
                              false,
                              false,
                              "unknown",
                              &to_octave_str_unknown,
                              &to_octave_bin_unknown,
                              &from_octave_str_unknown,
                              &from_octave_bin_unknown};

/* end type unknown */

// helpers for network types

static inline
int to_octave_bin_cidr_inet (const char *c, octave_value &ov, bool &cidr)
{
  int8_t nb;

  if (*(c++) == AF_INET)
    {
      uint8NDArray a (dim_vector (5, 1), 0);

      a(4) = uint8_t (*(c++)); // number of set mask bits

      cidr = *(c++);

      nb = *(c++);

      if (nb < 0)
        nb = 0;

      if (nb > 4)
        {
          c_verror ("internal error: received too many bytes for AF_INET type");

          return 1;
        }

      memcpy (a.fortran_vec (), c, nb);

      ov = octave_value (a);
    }
  else
    {
      uint16NDArray a (dim_vector (9, 1), 0);

      a(8) = uint16_t (uint8_t (*(c++))); // number of set mask bits

      cidr = *(c++);

      nb = *(c++);

      if (nb < 0)
        nb = 0;

      if (nb > 16)
        {
          c_verror ("internal error: received too many bytes for AF_INET6 type");

          return 1;
        }

      int8_t n = nb / 2;
      bool odd = nb % 2;

      for (int8_t i = 0; i < n; i++, c += 2)
        a(i) = uint16_t (be16toh (*((uint16_t *) c)));

      if (odd)
        {
          uint16_t tp = *c;

          a(n) = uint16_t (be16toh (tp));
        }

      ov = octave_value (a);
    }

  return 0;
}

static inline
int from_octave_bin_cidr_inet (const octave_value &ov, oct_pq_dynvec_t &val,
                               bool cidr)
{
  int nl = ov.numel ();

  uint8_t n_mbits;

  bool err;

  if (nl == 4 || nl == 5)
    {
      uint8NDArray a;
      SET_ERR (a = ov.uint8_array_value (), err);

      if (err)
        {
          c_verror ("can not convert octave_value to network type representation");
          return 1;
        }

      if (nl == 5)
        n_mbits = a(4).value ();
      else
        n_mbits = 32;

      val.push_back (AF_INET);
      val.push_back (n_mbits);
      val.push_back (cidr);
      val.push_back (4);
      octave_idx_type csize = val.size ();
      val.resize (csize + 4);
      memcpy (&(val[csize]), a.fortran_vec (), 4);
    }
  else if (nl == 8 || nl == 9)
    {
      uint16NDArray a;
      SET_ERR (a = ov.uint16_array_value (), err);

      if (err)
        {
          c_verror ("can not convert octave_value to network type representation");
          return 1;
        }

      if (nl == 9)
        n_mbits = a(8).value ();
      else
        n_mbits = 128;

      val.push_back (PGSQL_AF_INET6);
      val.push_back (n_mbits);
      val.push_back (cidr);
      val.push_back (16);
      for (int i = 0; i < 8; i++)
        {
          OCT_PQ_PUT(val, uint16_t, htobe16(a(i).value ()))
        }
    }
  else
    {
      c_verror ("invalid network type representation");
      return 1;
    }

  return 0;
}

// end helpers for network types

/* type cidr */

int to_octave_str_cidr (const octave_pq_connection_rep &conn,
                        const char *c, octave_value &ov, int nb)
{
  return 1;
}

int to_octave_bin_cidr (const octave_pq_connection_rep &conn,
                        const char *c, octave_value &ov, int nb)
{
  bool cidr = false;

  if (to_octave_bin_cidr_inet (c, ov, cidr))
    return 1;

  if (! cidr)
    {
      c_verror ("internal error: unexpected flag in cidr type");

      return 1;
    }

  return 0;
}

int from_octave_str_cidr (const octave_pq_connection_rep &conn,
                          const octave_value &ov, oct_pq_dynvec_t &val)
{
  return 1;
}

int from_octave_bin_cidr (const octave_pq_connection_rep &conn,
                          const octave_value &ov, oct_pq_dynvec_t &val)
{
  return from_octave_bin_cidr_inet (ov, val, true);
}

oct_pq_conv_t conv_cidr = {0,
                           0,
                           oct_pq_el_oids_t (),
                           oct_pq_conv_cache_t (),
                           false,
                           false,
                           false,
                           "cidr",
                           &to_octave_str_cidr,
                           &to_octave_bin_cidr,
                           &from_octave_str_cidr,
                           &from_octave_bin_cidr};

/* end type cidr */

/* type inet */

int to_octave_str_inet (const octave_pq_connection_rep &conn,
                        const char *c, octave_value &ov, int nb)
{
  return 1;
}

int to_octave_bin_inet (const octave_pq_connection_rep &conn,
                        const char *c, octave_value &ov, int nb)
{
  bool cidr = false;

  if (to_octave_bin_cidr_inet (c, ov, cidr))
    return 1;

  if (cidr)
    {
      c_verror ("internal error: unexpected flag in inet type");

      return 1;
    }

  return 0;
}

int from_octave_str_inet (const octave_pq_connection_rep &conn,
                          const octave_value &ov, oct_pq_dynvec_t &val)
{
  return 1;
}

int from_octave_bin_inet (const octave_pq_connection_rep &conn,
                          const octave_value &ov, oct_pq_dynvec_t &val)
{
  return from_octave_bin_cidr_inet (ov, val, false);
}

oct_pq_conv_t conv_inet = {0,
                           0,
                           oct_pq_el_oids_t (),
                           oct_pq_conv_cache_t (),
                           false,
                           false,
                           false,
                           "inet",
                           &to_octave_str_inet,
                           &to_octave_bin_inet,
                           &from_octave_str_inet,
                           &from_octave_bin_inet};

/* end type inet */

/* type macaddr */

int to_octave_str_macaddr (const octave_pq_connection_rep &conn,
                           const char *c, octave_value &ov, int nb)
{
  return 1;
}

int to_octave_bin_macaddr (const octave_pq_connection_rep &conn,
                           const char *c, octave_value &ov, int nb)
{
  uint8NDArray a (dim_vector (6, 1));

  memcpy (a.fortran_vec (), c, 6);

  ov = octave_value (a);

  return 0;
}

int from_octave_str_macaddr (const octave_pq_connection_rep &conn,
                             const octave_value &ov, oct_pq_dynvec_t &val)
{
  return 1;
}

int from_octave_bin_macaddr (const octave_pq_connection_rep &conn,
                             const octave_value &ov, oct_pq_dynvec_t &val)
{
  uint8NDArray a;
  bool err;
  SET_ERR (a = ov.uint8_array_value (), err);

  if (err)
    {
      c_verror ("can not convert octave_value to macaddr representation");
      return 1;
    }

  if (a.numel () != 6)
    {
      c_verror ("macaddr representation must have 6 elements");
      return 1;
    }

  octave_idx_type csize = val.size ();

  val.resize (csize + 6);

  memcpy (&(val[csize]), a.fortran_vec (), 6);

  return 0;
}

oct_pq_conv_t conv_macaddr = {0,
                              0,
                              oct_pq_el_oids_t (),
                              oct_pq_conv_cache_t (),
                              false,
                              false,
                              false,
                              "macaddr",
                              &to_octave_str_macaddr,
                              &to_octave_bin_macaddr,
                              &from_octave_str_macaddr,
                              &from_octave_bin_macaddr};

/* end type macaddr */

/* type bit */

int to_octave_str_bit (const octave_pq_connection_rep &conn,
                       const char *c, octave_value &ov, int nb)
{
  return 1;
}

int to_octave_bin_bit (const octave_pq_connection_rep &conn,
                       const char *c, octave_value &ov, int nb)
{
  int32_t nbits = int32_t (be32toh (*((int32_t *) c)));

  c += 4;

  int32_t nbytes = (nbits + 7) / 8;

  uint8NDArray a (dim_vector (nbytes, 1));

  memcpy (a.fortran_vec (), c, nbytes);

  octave_scalar_map tp;
  tp.assign ("bitlen", octave_value (octave_int32 (nbits)));
  tp.assign ("bits", octave_value (a));

  ov = octave_value (tp);

  return 0;
}

int from_octave_str_bit (const octave_pq_connection_rep &conn,
                         const octave_value &ov, oct_pq_dynvec_t &val)
{
  return 1;
}

int from_octave_bin_bit (const octave_pq_connection_rep &conn,
                         const octave_value &ov, oct_pq_dynvec_t &val)
{
  octave_scalar_map tp;
  bool err;
  SET_ERR (tp = ov.scalar_map_value (), err);
  if (err || ! tp.isfield ("bitlen") || ! tp.isfield ("bits"))
    {
      c_verror ("can not convert octave_value to bitstring representation");
      return 1;
    }

  int32_t nbits = 0;
  SET_ERR (nbits = tp.contents ("bitlen").int_value (), err);

  uint8NDArray a;
  if (! err)
    {
      SET_ERR (a = tp.contents ("bits").uint8_array_value (), err);
    }

  if (err || nbits < 0)
    {
      c_verror ("can not convert octave_value to bitstring representation");
      return 1;
    }

  int32_t nbytes = (nbits + 7) / 8;

  if (a.numel () != nbytes)
    {
      c_verror ("wrong number of elements in bitstring representation");
      return 1;
    }

  OCT_PQ_PUT(val, int32_t, htobe32 (nbits))

  octave_idx_type csize = val.size ();

  val.resize (csize + nbytes);

  memcpy (&(val[csize]), a.fortran_vec (), nbytes);

  return 0;
}

oct_pq_conv_t conv_bit = {0,
                          0,
                          oct_pq_el_oids_t (),
                          oct_pq_conv_cache_t (),
                          false,
                          false,
                          false,
                          "bit",
                          &to_octave_str_bit,
                          &to_octave_bin_bit,
                          &from_octave_str_bit,
                          &from_octave_bin_bit};

/* end type bit */

/* type varbit */

oct_pq_conv_t conv_varbit = {0,
                             0,
                             oct_pq_el_oids_t (),
                             oct_pq_conv_cache_t (),
                             false,
                             false,
                             false,
                             "varbit",
                             &to_octave_str_bit,
                             &to_octave_bin_bit,
                             &from_octave_str_bit,
                             &from_octave_bin_bit};

/* end type varbit */

/* type uuid */

int to_octave_str_uuid (const octave_pq_connection_rep &conn,
                        const char *c, octave_value &ov, int nb)
{
  return 1;
}

int to_octave_bin_uuid (const octave_pq_connection_rep &conn,
                        const char *c, octave_value &ov, int nb)
{
  uint8NDArray m (dim_vector (16, 1));

  memcpy (m.fortran_vec (), c, 16);

  ov = octave_value (m);

  return 0;
}

int from_octave_str_uuid (const octave_pq_connection_rep &conn,
                          const octave_value &ov, oct_pq_dynvec_t &val)
{
  return 1;
}

int from_octave_bin_uuid (const octave_pq_connection_rep &conn,
                          const octave_value &ov, oct_pq_dynvec_t &val)
{
  uint8NDArray b;
  bool err;
  SET_ERR (b = ov.uint8_array_value (), err);

  if (err)
    {
      c_verror ("can not convert octave_value to uuid representation");
      return 1;
    }

  if (b.numel () != 16)
    {
      c_verror ("uuid representation must have 16 elements");
      return 1;
    }

  octave_idx_type csize = val.size ();

  val.resize (csize + 16);

  memcpy (&(val[csize]), b.fortran_vec (), 16);

  return 0;
}

oct_pq_conv_t conv_uuid = {0,
                           0,
                           oct_pq_el_oids_t (),
                           oct_pq_conv_cache_t (),
                           false,
                           false,
                           false,
                           "uuid",
                           &to_octave_str_uuid,
                           &to_octave_bin_uuid,
                           &from_octave_str_uuid,
                           &from_octave_bin_uuid};

/* end type uuid */

/* type xml */

int to_octave_str_xml (const octave_pq_connection_rep &conn,
                       const char *c, octave_value &ov, int nb)
{
  return 1;
}

int to_octave_bin_xml (const octave_pq_connection_rep &conn,
                       const char *c, octave_value &ov, int nb)
{
  std::string s (c, nb);

  ov = octave_value (s);

  return 0;
}

int from_octave_str_xml (const octave_pq_connection_rep &conn,
                         const octave_value &ov, oct_pq_dynvec_t &val)
{
  return 1;
}

int from_octave_bin_xml (const octave_pq_connection_rep &conn,
                         const octave_value &ov, oct_pq_dynvec_t &val)
{
  std::string s;
  bool err;
  SET_ERR (s = ov.string_value (), err);

  if (err)
    {
      c_verror ("can not convert octave_value to string");
      return 1;
    }

  octave_idx_type l = s.size ();

  octave_idx_type csize = val.size ();

  val.resize (csize + l);

  memcpy (&(val[csize]), s.data (), l);

  return 0;
}

oct_pq_conv_t conv_xml = {0,
                          0,
                          oct_pq_el_oids_t (),
                          oct_pq_conv_cache_t (),
                          false,
                          false,
                          false,
                          "xml",
                          &to_octave_str_text,
                          &to_octave_bin_text,
                          &from_octave_str_text,
                          &from_octave_bin_text};

/* end type xml */

/* type record */

int to_octave_str_record (const octave_pq_connection_rep &conn,
                          const char *c, octave_value &ov, int nb)
{
  return 1;
}

int to_octave_bin_record (const octave_pq_connection_rep &conn,
                          const char *v, octave_value &ov, int nb)
{
  const char *p = v;

  // ncols
  OCT_PQ_DECL_GET_INT32(nl, p, int32_t)

  // elements
  Cell c (nl, 1);
  for (int i = 0; i < nl; i++)
    {
      // element OID
      OCT_PQ_DECL_GET_INT32(oid, p, uint32_t)

      OCT_PQ_DECL_GET_INT32(null_id, p, int32_t)

      if (null_id == -1)
        // NULL
        c(i) = octave_value (octave_NA);
      else
        {
          uint32_t nb_el = uint32_t (null_id);

          oct_pq_conv_t *el_conv;
          pq_oct_type_t oct_type;

          if (! (el_conv = pgtype_from_spec (conn, oid, oct_type)))
            return 1;

          octave_value el;
          switch (oct_type)
            {
            case simple:
              if (el_conv->to_octave_bin (conn, p, el, nb_el))
                return 1;
              break;

            case array:
              if (to_octave_bin_array (conn, p, el, nb_el, el_conv))
                return 1;
              break;

            case composite:
              if (to_octave_bin_composite (conn, p, el, nb_el, el_conv))
                return 1;
              break;

            default:
              // should not get here
              c_verror ("'record' converter: internal error, undefined type identifier");
              return 1;
            }

          p += nb_el;

          c(i) = el;
        }
    }

  ov = octave_value (c);

  return 0;
}

int from_octave_str_record (const octave_pq_connection_rep &conn,
                            const octave_value &ov, oct_pq_dynvec_t &val)
{
  return 1;
}

int from_octave_bin_record (const octave_pq_connection_rep &conn,
                            const octave_value &ov, oct_pq_dynvec_t &val)
{
  c_verror ("Type 'record' can't be sent to postgresql.");

  return 1;
}

oct_pq_conv_t conv_record = {0,
                             0,
                             oct_pq_el_oids_t (),
                             oct_pq_conv_cache_t (),
                             false,
                             false,
                             false,
                             "record",
                             &to_octave_str_record,
                             &to_octave_bin_record,
                             &from_octave_str_record,
                             &from_octave_bin_record};

/* end type record */

oct_pq_conv_t *t_conv_ptrs[OCT_PQ_NUM_CONVERTERS] = {&conv_bool,
                                                     &conv_oid,
                                                     &conv_float8,
                                                     &conv_float4,
                                                     &conv_text,
                                                     &conv_varchar,
                                                     &conv_bpchar,
                                                     &conv_name,
                                                     &conv_bytea,
                                                     &conv_int2,
                                                     &conv_int4,
                                                     &conv_int8,
                                                     &conv_money,
                                                     &conv_timestamp,
                                                     &conv_timestamptz,
                                                     &conv_interval,
                                                     &conv_time,
                                                     &conv_timetz,
                                                     &conv_date,
                                                     &conv_point,
                                                     &conv_lseg,
                                                     &conv_line,
                                                     &conv_box,
                                                     &conv_circle,
                                                     &conv_polygon,
                                                     &conv_path,
                                                     &conv_unknown,
                                                     &conv_cidr,
                                                     &conv_inet,
                                                     &conv_macaddr,
                                                     &conv_bit,
                                                     &conv_varbit,
                                                     &conv_uuid,
                                                     &conv_xml,
                                                     &conv_record};

oct_pq_conv_ptrs_t conv_ptrs (OCT_PQ_NUM_CONVERTERS, t_conv_ptrs);
