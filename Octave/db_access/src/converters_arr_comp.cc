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
#include <octave/Cell.h>
#include <octave/lo-ieee.h>

#include <stdint.h>

#include "converters.h"
#include "pq_connection.h"
#include "error-helpers.h"

#define ERROR_RETURN_NO_PG_TYPE                                         \
 {                                                                      \
   c_verror ("could not determine postgresql type for Octave parameter"); \
   return NULL;                                                         \
 }

oct_pq_conv_t *pgtype_from_spec (const octave_pq_connection_rep &conn,
                                 std::string &name,
                                 pq_oct_type_t &oct_type)
{
  oct_pq_conv_t *conv = NULL;

  // printf ("pgtype_from_spec(%s): simple ", name.c_str ());

  oct_type = simple; // default
  int l;
  while (name.size () >= 2 && ! name.compare (l = name.size () - 2, 2, "[]"))
    {
      name.erase (l, 2);
      oct_type = array;

      // printf ("array ");
    }

  oct_pq_name_conv_map_t::const_iterator iter;

  if ((iter = conn.name_conv_map.find (name.c_str ())) ==
      conn.name_conv_map.end ())
    {
      c_verror ("no converter found for type %s",
                name.c_str ());
      return NULL;
    }
  else
    {
      // printf ("(looked up in name map) ");

      conv = iter->second.get_copy ();

      if (oct_type == array && ! conv->aoid)
        {
          c_verror ("%s: internal error, type %s, specified as array, has no array type in system catalog", name.c_str ());
          return NULL;
        }

      if (! (oct_type == array) && conv->is_composite)
        {
          oct_type = composite;

          // printf ("composite ");
        }
    }

  // printf ("\n");

  return conv;
}

oct_pq_conv_t *pgtype_from_spec (const octave_pq_connection_rep &conn, Oid oid,
                                 pq_oct_type_t &oct_type)
{
  // printf ("pgtype_from_spec(%u): ", oid);

  oct_pq_conv_t *conv = NULL;

  oct_pq_conv_map_t::const_iterator iter;
  
  if ((iter = conn.conv_map.find (oid)) == conn.conv_map.end ())
    {
      c_verror ("no converter found for element oid %u", oid);
      return NULL;
    }
  conv = iter->second.get_copy ();
  // printf ("(looked up %s in oid map) ", conv->name.c_str ());

  if (conv->aoid == oid)
    oct_type = array;
  else if (conv->is_composite)
    oct_type = composite;
  else
    oct_type = simple;

  // printf ("oct_type: %i\n", oct_type);

  return conv;
}

oct_pq_conv_t *pgtype_from_spec (const octave_pq_connection_rep &conn, Oid oid,
                                 oct_pq_conv_t *&c_conv,
                                 pq_oct_type_t &oct_type)
{
  if (c_conv)
    {
      if (c_conv->aoid == oid)
        oct_type = array;
      else if (c_conv->is_composite)
        oct_type = composite;
      else
        oct_type = simple;
    }
  else
    c_conv = pgtype_from_spec (conn, oid, oct_type);

  return c_conv;
}

oct_pq_conv_t *pgtype_from_octtype (const octave_pq_connection_rep &conn,
                                    const octave_value &param)
{
  // printf ("pgtype_from_octtype: ");

  if (param.is_bool_scalar ())
    {
      // printf ("bool\n");
      return conn.name_conv_map.find ("bool")->second.get_copy ();
    }
  else if (param.is_real_scalar ())
    {
      if (param.is_double_type ())
        {
          // printf ("float8\n");
          return conn.name_conv_map.find ("float8")->second.get_copy ();
        }
      else if (param.is_single_type ())
        {
          // printf ("float4\n");
          return conn.name_conv_map.find ("float4")->second.get_copy ();
        }
    }

  if (param.is_scalar_type ())
    {
      if (param.is_int16_type ())
        {
          // printf ("int2\n");
          return conn.name_conv_map.find ("int2")->second.get_copy ();
        }
      else if (param.is_int32_type ())
        {
          // printf ("int4\n");
          return conn.name_conv_map.find ("int4")->second.get_copy ();
        }
      else if (param.is_int64_type ())
        {
          // printf ("int8\n");
          return conn.name_conv_map.find ("int8")->second.get_copy ();
        }
      else if (param.is_uint32_type ())
        {
          // printf ("oid\n");
          return conn.name_conv_map.find ("oid")->second.get_copy ();
        }
    }

  if (param.is_uint8_type ())
    {
      // printf ("bytea\n");
      return conn.name_conv_map.find ("bytea")->second.get_copy ();
    }
  else if (param.is_string ())
    {
      // printf ("text\n");
      return conn.name_conv_map.find ("text")->second.get_copy ();
    }

  // is_real_type() is true for strings, so is_numeric_type() would
  // still be needed if strings were not recognized above
  if (param.is_real_type ())
    {
      switch (param.numel ())
        {
        case 2:
          // printf ("point\n");
          return conn.name_conv_map.find ("point")->second.get_copy ();
        case 3:
          // printf ("circle\n");
          return conn.name_conv_map.find ("circle")->second.get_copy ();
        case 4:
          // printf ("lseg\n");
          return conn.name_conv_map.find ("lseg")->second.get_copy ();
        }
    }

  ERROR_RETURN_NO_PG_TYPE
}

octave_idx_type count_row_major_order (dim_vector &dv,
                                       oct_mo_count_state &state, bool init)
{
  if (init)
    {
      state.nd = dv.length ();

      state.cur.resize (state.nd);

      state.pd.resize (state.nd);

      // cannot be done with resize in multiple initializations
      // (resizing to n<2 not possible) and there is no fill() method
      for (octave_idx_type i = 0; i < state.nd; i++)
        {
          state.cur(i) = 0;
          state.pd(i) = 1;
        }

      for (octave_idx_type i = 0; i < state.nd - 1; i++)
        {
           state.pd(i + 1) = state.pd(i) * dv(i);
        }

      return 0;
    }
  else
    {
      octave_idx_type nd = state.nd;

      if (nd > 0)
        {
          octave_idx_type ret = 0;

          state.cur(nd - 1)++;

          for (octave_idx_type i = nd - 1; i > 0; i--)
            {
              if (state.cur(i) == dv(i))
                {
                  state.cur(i) = 0;
                  state.cur(i - 1)++;
                }

              ret += state.cur(i) * state.pd(i);
            }

          if (state.cur(0) == dv(0))
            ret = -1; // signals overflow
          else
            ret += state.cur(0) * state.pd(0);

          return ret;
        }
      else
        return -1;
    }
}

int from_octave_bin_array (const octave_pq_connection_rep &conn,
                           const octave_value &oct_arr,
                           oct_pq_dynvec_t &val, oct_pq_conv_t *conv)
{
  octave_scalar_map m;
  bool err;
  SET_ERR (m= oct_arr.scalar_map_value (), err);
  if (err)
    {
      c_verror ("Postgresql array parameter no Octave structure");
      return 1;
    }

  if (! m.isfield ("ndims") || ! m.isfield ("data"))
    {
      c_verror ("field 'ndims' or 'data' missing in parameter for Postgresql array");
      return 1;
    }

  octave_idx_type nd_pq = 0;
  SET_ERR (nd_pq = m.contents ("ndims").int_value (), err);

  Cell arr;
  if (! err)
    {
      SET_ERR (arr = m.contents ("data").cell_value (), err);
    }
  if (err || nd_pq < 0)
    {
      c_verror ("'ndims' and 'data' could not be converted to non-negative integer and cell-array in parameter for Postgresql array");
      return 1;
    }

  RowVector lb;

  // Are lbounds given?
  if (m.isfield ("lbounds"))
    {
      SET_ERR (lb = m.contents ("lbounds").row_vector_value (), err);
      if (err)
        {
          c_verror ("could not convert given enumeration bases for array to row vector");
          return 1;
        }
    }

  octave_idx_type nl = arr.numel ();

  // dimensions of Octaves representation of postgresql array
  dim_vector d = arr.dims ();
  d.chop_trailing_singletons ();
  int nd_oct = d.length ();
  if (nd_oct == 2 && d(1) == 1)
    nd_oct = 1;

  // check dimensions
  if (nd_oct > nd_pq)
    {
      c_verror ("given representation of postgresql array has more dimensions than specified");
      return 1;
    }

  // check lbounds
  if (nd_pq > 0 && lb.is_empty ())
    lb.resize (nd_pq, 1); // fill with 1
  else if (lb.numel () != nd_pq)
    {
      c_verror ("number of specified enumeration bases for array does not match specified number of dimensions");
      return 1;
    }

  // ndim
  OCT_PQ_PUT(val, int32_t, htobe32 ((int32_t) nd_pq))
  // always make a NULL-bitmap, we don't scan for NULLs to see if it
  // is really necessary
  OCT_PQ_PUT(val, int32_t, htobe32 ((int32_t) 1))
  // element OID
  OCT_PQ_PUT(val, uint32_t, htobe32 ((uint32_t) conv->oid))
  // dims, lbounds
  for (int i = 0; i < nd_pq; i++)
    {
      OCT_PQ_PUT(val, int32_t, htobe32 ((int32_t) (i < nd_oct ? d(i) : 1)))
      OCT_PQ_PUT(val, int32_t, htobe32 ((int32_t) lb(i)))
    }
  // elements
  oct_mo_count_state state;
  count_row_major_order (d, state, true); // initialize counter
  for (int i = 0, ti = 0; ti < nl;
       ti++, i = count_row_major_order (d, state, false))
    {
      if (arr(i).is_real_scalar () && arr(i).isna ().bool_value ())
        // a length value (uint32_t) would not have the 6 highest bits
        // set (while this has all bits set)
        { OCT_PQ_PUT(val, int32_t, htobe32 ((int32_t) -1)) }
      else
        {
          OCT_PQ_SET_UINT32_PLACEHOLDER(val, temp_pos)

            if (conv->is_composite)
              {
                if (from_octave_bin_composite (conn, arr(i), val, conv))
                  return 1;
              }
            else
              {
                if (conv->from_octave_bin (conn, arr(i), val))
                  return 1;
              }

          OCT_PQ_FILL_UINT32_PLACEHOLDER(val, temp_pos)
        }
    }

  return 0;
}

int from_octave_bin_composite (const octave_pq_connection_rep &conn,
                               const octave_value &oct_comp,
                               oct_pq_dynvec_t &val,
                               oct_pq_conv_t *conv)
{
  Cell rec;
  bool err;
  SET_ERR (rec = oct_comp.cell_value (), err);
  if (err)
    {
      c_verror ("Octaves representation of a composite type could not be converted to cell-array");
      return 1;
    }

  octave_idx_type nl = rec.numel ();

  if (size_t (nl) != conv->el_oids.size ())
    {
      c_verror ("Octaves representation of a composite type has incorrect number of elements (%i, should have %i)",
             nl, conv->el_oids.size ());

      return 1;
    }

  // ncols
  OCT_PQ_PUT(val, int32_t, htobe32 ((int32_t) nl))
  // elements
  for (int i = 0; i < nl; i++)
    {
      // element OID
      OCT_PQ_PUT(val, uint32_t, htobe32 ((uint32_t) conv->el_oids[i]))
      if (rec(i).is_real_scalar () && rec(i).isna ().bool_value ())
        // a length value (uint32_t) would not have the 6 highest bits
        // set (while this has all bits set)
        { OCT_PQ_PUT(val, int32_t, htobe32 ((int32_t) -1)) }
      else
        {
          OCT_PQ_SET_UINT32_PLACEHOLDER(val, temp_pos)

          oct_pq_conv_t *el_conv;
          pq_oct_type_t oct_type;

          if (! (el_conv = pgtype_from_spec (conn,
                                             conv->el_oids[i],
                                             conv->conv_cache[i],
                                             oct_type)))
            return 1;

          switch (oct_type)
            {
            case simple:
              if (el_conv->from_octave_bin (conn, rec(i), val))
                return 1;
              break;

            case array:
              if (from_octave_bin_array (conn, rec(i), val, el_conv))
                return 1;
              break;

            case composite:
              if (from_octave_bin_composite (conn, rec(i), val, el_conv))
                return 1;
              break;

            default:
              // should not get here
              c_verror ("internal error, undefined type identifier");
              return 1;
            }

          OCT_PQ_FILL_UINT32_PLACEHOLDER(val, temp_pos)
        }
    }

  return 0;
}

int from_octave_str_array (const octave_pq_connection_rep &conn,
                           const octave_value &oct_arr,
                           oct_pq_dynvec_t &val, octave_value &type)
{
  // not implemented
  c_verror ("not implemented");
  return 1;

  return 0;
}

int from_octave_str_composite (const octave_pq_connection_rep &conn,
                               const octave_value &oct_comp,
                               oct_pq_dynvec_t &val,
                               octave_value &type)
{
  // not implemented
  c_verror ("not implemented");
  return 1;

  return 0;
}

int to_octave_bin_array (const octave_pq_connection_rep &conn,
                         const char *v, octave_value &ov, int nb,
                         oct_pq_conv_t *conv)
{
  const char *p = v;

  // ndim
  OCT_PQ_DECL_GET_INT32(ndim, p, int32_t)
  // Has NULL bitmap. Since this information is not used, comment it
  // out to avoid a warning and increase the pointer manually.
  //
  // OCT_PQ_DECL_GET_INT32(hasnull, p, int32_t)
  p += 4;
  //
  // element OID
  OCT_PQ_DECL_GET_INT32(oid, p, uint32_t)

  // check element OID
  if (oid != conv->oid)
    {
      c_verror ("element oid %i sent by server does not match element oid %i expected for array with oid %i",
             oid, conv->oid, conv->aoid);
      return 1;
    }

  // dims, lbounds
  dim_vector dv;
  dv.resize (ndim, 0); // ndim < 2 is treated as ndim == 2
  if (ndim == 1)
    dv(1) = 1;

  RowVector lbounds (ndim);

  for (int i = 0; i < ndim; i++)
    {
      OCT_PQ_GET_INT32(dv(i), p, int32_t)
      OCT_PQ_GET_INT32(lbounds(i), p, int32_t)
    }
  //

  // elements
  octave_idx_type nl = dv.numel ();
  Cell c (dv);
  oct_mo_count_state state;
  count_row_major_order (dv, state, true); // initialize counter
  for (int i = 0, ti = 0; ti < nl;
       ti++, i = count_row_major_order (dv, state, false))
    {
      OCT_PQ_DECL_GET_INT32(null_id, p, int32_t)
      if (null_id == -1)
        // NULL
        c(i) = octave_value (octave_NA);
      else
        {
          uint32_t nb_el = uint32_t (null_id);

          octave_value ov_el;

          if (conv->is_composite)
            {
              if (to_octave_bin_composite (conn, p, ov_el, nb_el, conv))
                return 1;
            }
          else
            {
              if (conv->to_octave_bin (conn, p, ov_el, nb_el))
                return 1;
            }

          p += nb_el;

          c(i) = ov_el;
        }
    }

  octave_scalar_map m;
  m.assign ("data", octave_value (c));
  m.assign ("ndims", octave_value (ndim));
  m.assign ("lbounds", octave_value (lbounds));

  ov = octave_value (m);

  return 0;
}

int to_octave_bin_composite (const octave_pq_connection_rep &conn,
                             const char *v, octave_value &ov, int nb,
                             oct_pq_conv_t *conv)
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

          if (! (el_conv = pgtype_from_spec (conn, oid, conv->conv_cache[i],
                                             oct_type)))
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
              c_verror ("internal error, undefined type identifier");
              return 1;
            }

          p += nb_el;

          c(i) = el;
        }
    }

  ov = octave_value (c);

  return 0;
}


int to_octave_str_array (const octave_pq_connection_rep &conn,
                         const char *v, octave_value &ov, int nb,
                         oct_pq_conv_t *conv)
{
  // not implemented
  c_verror ("not implemented");
  return 1;

  return 0;
}

int to_octave_str_composite (const octave_pq_connection_rep &conn,
                             const char *v, octave_value &ov, int nb,
                             oct_pq_conv_t *conv)
{
  // not implemented
  c_verror ("not implemented");
  return 1;

  return 0;
}
