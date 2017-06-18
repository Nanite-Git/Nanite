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

#include "pq_connection.h"
#include "command.h"
#include "error-helpers.h"

DEFINE_OV_TYPEID_FUNCTIONS_AND_DATA (octave_pq_connection, "PGconn", "PGconn")

std::string &pq_basetype_prefix (void)
{
  static std::string prefix ("pg_catalog.");
  return prefix;
}

const int pq_bpl = pq_basetype_prefix ().size ();

static bool map_str_cmp (const char *c1, const char *c2)
{
  if (strcmp (c1, c2) < 0)
    return true;
  else
    return false;
}

static bool map_string_cmp (const std::string &s1, const std::string &s2)
{
  if (s1.compare (s2) < 0)
    return true;
  else
    return false;
}

octave_pq_connection_rep::octave_pq_connection_rep (std::string &arg)
: conv_map (), name_conv_map (&map_str_cmp), conn (NULL), postgres (0)
{
  BEGIN_INTERRUPT_IMMEDIATELY_IN_FOREIGN_CODE;
  conn = PQconnectdb (arg.c_str ());
  END_INTERRUPT_IMMEDIATELY_IN_FOREIGN_CODE;

  if (! conn || PQstatus (conn) == CONNECTION_BAD)
    {
      if (conn)
        {
          c_verror ("%s", PQerrorMessage (conn));

          PGconn *t_conn = conn;

          conn = NULL;

          BEGIN_INTERRUPT_IMMEDIATELY_IN_FOREIGN_CODE;
          PQfinish (t_conn);
          END_INTERRUPT_IMMEDIATELY_IN_FOREIGN_CODE;
        }

      c_verror ("PQ connection attempt failed");
    }
  else
    {
      // init name converter-map (kind of "bootstrapping")
      for (int i = 0; i < OCT_PQ_NUM_CONVERTERS; i++)
        {
          name_conv_map[conv_ptrs[i]->name.c_str ()] = conv_ptrs[i];

          // unqualified name, may be replaced later with user-defined type
          name_conv_map[conv_ptrs[i]->name.c_str () + pq_bpl] = conv_ptrs[i];
        }

      if (octave_pq_get_postgres_oid () ||
          octave_pq_fill_base_types () ||
          octave_pq_get_composite_types () ||
          octave_pq_get_enum_types ())
        {
          PGconn *t_conn = conn;

          conn = NULL;

          BEGIN_INTERRUPT_IMMEDIATELY_IN_FOREIGN_CODE;
          PQfinish (t_conn);
          END_INTERRUPT_IMMEDIATELY_IN_FOREIGN_CODE;

          c_verror ("could not read types");
        }
      else
        {
          if (strcmp (PQparameterStatus (conn, "integer_datetimes"), "on"))
            integer_datetimes = false;
          else
            integer_datetimes = true;
        }
    }
}

octave_pq_connection_rep::~octave_pq_connection_rep (void)
{
  if (conn)
    {
      BEGIN_INTERRUPT_IMMEDIATELY_IN_FOREIGN_CODE;
      PQfinish (conn);
      END_INTERRUPT_IMMEDIATELY_IN_FOREIGN_CODE;

      octave_pq_delete_non_constant_types ();
    }
}

void octave_pq_connection_rep::octave_pq_close (void)
{
  if (conn)
    {
      PGconn *t_conn = conn;

      conn = NULL;

      BEGIN_INTERRUPT_IMMEDIATELY_IN_FOREIGN_CODE;
      PQfinish (t_conn);
      END_INTERRUPT_IMMEDIATELY_IN_FOREIGN_CODE;

      octave_pq_delete_non_constant_types ();
    }
  else
    // deliberately left the 'error' call here, since
    // 'octave_pq_close()' is only called by 'pq_close' immediately
    // before returning
    error ("PGconn object not open");
}

void octave_pq_connection_rep::octave_pq_delete_non_constant_types (void)
{
  // In the first map, allocated types are usually referenced twice
  // (by oid and aoid). Yet we need no refcount as long as we go
  // through the name map as the last, since there (the same) types
  // are only referenced once.

  std::vector<oct_pq_conv_map_t::iterator> t_it_v;

  for (oct_pq_conv_map_t::iterator it = conv_map.begin ();
       it != conv_map.end ();
       it++)
    if (it->second->is_not_constant)
      t_it_v.push_back (it);

  for (std::vector<oct_pq_conv_map_t::iterator>::iterator it = t_it_v.begin ();
       it != t_it_v.end (); it++)
    conv_map.erase (*it);

  std::vector<oct_pq_name_conv_map_t::iterator> t_name_it_v;

  for (oct_pq_name_conv_map_t::iterator it = name_conv_map.begin ();
       it != name_conv_map.end ();
       it++)
    {
      oct_pq_conv_t *conv = it->second;

      if (conv->is_not_constant)
        {
          t_name_it_v.push_back (it);

          delete conv;
        }
    }

  for (std::vector<oct_pq_name_conv_map_t::iterator>::iterator it =
         t_name_it_v.begin ();
       it != t_name_it_v.end (); it++)
    name_conv_map.erase (*it);
}

int octave_pq_connection_rep::octave_pq_get_postgres_oid (void)
{
  Cell p, pt, rt (1, 1);

  rt(0) = octave_value ("oid");

  std::string cmd ("select oid from pg_roles where rolname = 'postgres';"),
    caller ("octave_pq_get_postgres_oid");

  command c (*this, cmd, p, pt, rt, caller);
  if (! c.good ())
    {
      c_verror ("could not read pg_roles");
      return 1;
    }

  octave_value res = c.process_single_result ();
  if (! c.good ())
    return 1;

  bool err;
  SET_ERR (postgres = res.scalar_map_value ().contents ("data").cell_value ()(0).int_value (), err);
  if (err)
    return 1;

  return 0;
}

int octave_pq_connection_rep::octave_pq_fill_base_types (void)
{
  // assert postgres oid had been determined
  if (! postgres.int_value ()) return 1;

  Cell p (1, 1), pt (1, 1), rt (3, 1);
  p(0) = postgres;
  pt(0) = octave_value ("oid");
  rt(0) = octave_value ("oid");
  rt(1) = octave_value ("name");
  rt(2) = octave_value ("oid");

  std::string cmd ("select oid, typname, typarray from pg_type where (typowner = $1 AND typtype = 'b' AND typarray != 0) OR typname = 'record' OR typname = 'unknown';"),
    caller ("octave_pq_fill_base_types");

  command c (*this, cmd, p, pt, rt, caller);
  if (! c.good ())
    {
      c_verror ("octave_pq_fill_base_types: could not read pg_type");
      return 1;
    }

  octave_value res = c.process_single_result ();
  if (! c.good ())
    return 1;

  Cell tpls;
  bool err;
  SET_ERR (tpls = res.scalar_map_value ().contents ("data").cell_value (),
           err);
  if (err)
    {
      c_verror
        ("octave_pq_fill_base_types: could not convert result data to cell");
      return 1;
    }

  // make a temporary map of server base types (cell row numbers) for searching
  typedef std::map<std::string, int, bool (*) (const std::string &,
                                               const std::string &)>
    bt_map_t;
  bt_map_t bt_map (&map_string_cmp);
  for (int i = 0; i < tpls.rows (); i++)
    {
      SET_ERR (bt_map[tpls(i, 1).string_value ()] = i, err);
      if (err)
        break;
    }
  if (err)
    {
      c_verror ("octave_pq_fill_base_types: could not read returned result");
      return 1;
    }

  for (int i = 0; i < OCT_PQ_NUM_CONVERTERS; i++)
    {
      bt_map_t::iterator bt_it;
      if ((bt_it = bt_map.find (conv_ptrs[i]->name.c_str () + pq_bpl)) ==
          bt_map.end ())
        {
          c_verror ("octave_pq_fill_base_types: type %s not found in pg_type",
                    conv_ptrs[i]->name.c_str () + pq_bpl);
          return 1;
        }
      // fill in oid and aoid into static records of converters
      SET_ERR (conv_ptrs[i]->oid = tpls(bt_it->second, 0).int_value (), err);
      if (! err)
        {
          SET_ERR (conv_ptrs[i]->aoid = tpls(bt_it->second, 2).int_value (),
                   err);
        }
      if (err)
        break;

      // fill in map of converters over oid with oid and, if not zero,
      // also with aoid
      conv_map[conv_ptrs[i]->oid] = conv_ptrs[i];
      if (conv_ptrs[i]->aoid != 0)
        conv_map[conv_ptrs[i]->aoid] = conv_ptrs[i];
    }
  if (err)
    {
      c_verror ("octave_pq_fill_base_types: could not read returned result");
      return 1;
    }

  return 0;
}

int octave_pq_connection_rep::octave_pq_get_composite_types (void)
{
  Cell p, pt, rt;

  std::string cmd ("select pg_type.oid, pg_type.typname, pg_type.typarray, pg_namespace.nspname, pg_type_is_visible(pg_type.oid) as visible, array_agg(pg_attribute.atttypid), array_agg(pg_attribute.attnum) from (pg_type join pg_namespace on pg_type.typnamespace = pg_namespace.oid) join pg_attribute on pg_type.typrelid = pg_attribute.attrelid where pg_type.typtype = 'c' and pg_attribute.attnum > 0 group by pg_type.oid, pg_type.typname, pg_type.typarray, pg_type.typrelid, pg_namespace.nspname, visible;"),
    caller ("octave_pq_get_composite_types");

  command c (*this, cmd, p, pt, rt, caller);
  if (! c.good ())
    {
      c_verror ("octave_pq_get_composite_types: could not read pg_type");
      return 1;
    }

  octave_value res = c.process_single_result ();
  if (! c.good ())
    return 1;

  Cell tpls;
  bool err;
  SET_ERR (tpls = res.scalar_map_value ().contents ("data").cell_value (),
           err)
  if (err)
    {
      c_verror ("octave_pq_get_composite_types: could not convert result data to cell");
      return 1;
    }

  for (int i = 0; i < tpls.rows (); i++)
    {
      Oid oid;
      SET_ERR (oid = tpls(i, 0).uint_value (), err);
      Oid aoid;
      if (! err)
        {
          SET_ERR (aoid = tpls(i, 2).uint_value (), err);
        }
      std::string name;
      if (! err)
        {
          SET_ERR (name = tpls(i, 1).string_value (), err);
        }
      std::string nspace;
      if (! err)
        {
          SET_ERR (nspace = tpls(i, 3).string_value (), err);
        }
      bool visible = false;
      if (! err)
        {
          SET_ERR (visible = tpls(i, 4).bool_value (), err);
        }
      Cell r_el_oids;
      if (! err)
        {
          SET_ERR (r_el_oids = tpls(i, 5).scalar_map_value ()
                   .contents ("data").cell_value (), err);
        }
      Cell r_el_pos;
      if (! err)
        {
          SET_ERR (r_el_pos = tpls(i, 6).scalar_map_value ()
                   .contents ("data").cell_value (), err);
        }
      if (err)
        {
          c_verror ("octave_pq_get_composite_types: could not read returned result");
          return 1;
        }
      octave_idx_type nel = r_el_oids.numel ();
      if (nel != r_el_pos.numel ())
        {
          c_verror ("octave_pq_get_composite_types: internal error, inconsistent content of pg_attribute?");

          return 1;
        }
      oct_pq_el_oids_t el_oids;
      el_oids.resize (nel);
      oct_pq_conv_cache_t conv_cache;
      conv_cache.resize (nel);
      for (octave_idx_type i = 0; i < nel; i++)
        {
          octave_idx_type pos;
          // "column" number (attnum) is one-based, so subtract 1
          SET_ERR (pos = r_el_pos(i).idx_type_value () - 1, err);
          if (! err)
            {
              if (pos >= nel)
                {
                  c_verror ("octave_pq_get_composite_types: internal error (?system catalog erroneous?): column position %i greater than ncols %i for type %s, namespace %s",
                         pos, nel, name.c_str (), nspace.c_str ());
                  return 1;
                }

              SET_ERR (el_oids[pos] = r_el_oids(i).uint_value (), err);

              conv_cache[pos] = NULL;
            }
          if (err)
            {
              c_verror ("octave_pq_get_composite_types: could not fill in element oids.");

              return 1;
            }
        }

      // must be allocated and filled before creating the name map
      // entry, to get a remaining location for the c-string used as
      // key
      oct_pq_conv_t *t_conv = new oct_pq_conv_t;
      t_conv->oid = oid;
      t_conv->aoid = aoid;
      t_conv->el_oids = el_oids;
      t_conv->conv_cache = conv_cache;
      t_conv->is_composite = true;
      t_conv->is_enum = false;
      t_conv->is_not_constant = true;
      t_conv->name = nspace.append (".").append (name);
      t_conv->to_octave_str = NULL;
      t_conv->to_octave_bin = NULL;
      t_conv->from_octave_str = NULL;
      t_conv->from_octave_bin = NULL;

      oct_pq_conv_t *&by_oid = conv_map[oid],
        *&by_name = name_conv_map[t_conv->name.c_str ()];
      if (by_oid || by_name)
        {
          c_verror ("octave_pq_get_composite_types: internal error, key already in typemap (by_oid: %u/%li, by name: %s/%li)",
                 oid, by_oid, t_conv->name.c_str (), by_name);
          if (! by_oid) conv_map.erase (oid);
          if (! by_name) name_conv_map.erase (t_conv->name.c_str ());
          delete t_conv;
          return 1;
        }

      by_oid = by_name = t_conv;

      oct_pq_conv_t *t_conv_v = NULL; // silence inadequate warning by
                                      // initializing it here

      if (visible)
        {
          t_conv_v = new oct_pq_conv_t (*t_conv);

          t_conv_v->el_oids = el_oids;

          t_conv_v->conv_cache = conv_cache;

          t_conv_v->name = name;

          name_conv_map[t_conv_v->name.c_str ()] = t_conv_v;;
        }

      if (aoid)
        {
          oct_pq_conv_t *&by_aoid = conv_map[aoid];
          if (by_aoid)
            {
              c_verror ("octave_pq_get_composite_types: internal error, aoid key %u already in typemap", aoid);
              conv_map.erase (oid);
              name_conv_map.erase (t_conv->name.c_str ());
              delete t_conv;
              if (visible)
                {
                  name_conv_map.erase (t_conv_v->name.c_str ());
                  delete t_conv_v;
                }
              return 1;
            }

          by_aoid = by_oid;
        }

    }

  return 0;
}

int octave_pq_connection_rep::octave_pq_get_enum_types (void)
{
  Cell p, pt, rt;

  std::string cmd ("select pg_type.oid, pg_type.typname, pg_type.typarray, pg_namespace.nspname, pg_type_is_visible(pg_type.oid) from pg_type join pg_namespace on pg_type.typnamespace = pg_namespace.oid where pg_type.typtype = 'e';"),
    caller ("octave_pq_get_enum_types");

  command c (*this, cmd, p, pt, rt, caller);
  if (! c.good ())
    {
      c_verror ("octave_pq_get_enum_types: could not read pg_type");
      return 1;
    }
 
  octave_value res = c.process_single_result ();
  if (! c.good ())
    return 1;

  Cell tpls;
  bool err;
  SET_ERR (tpls = res.scalar_map_value ().contents ("data").cell_value (),
           err);
  if (err)
    {
      c_verror ("octave_pq_get_enum_types: could not convert result data to cell");
      return 1;
    }

  for (int i = 0; i < tpls.rows (); i++)
    {
      Oid oid;
      SET_ERR (oid = tpls(i, 0).uint_value (), err);
      Oid aoid;
      if (! err)
        {
          SET_ERR (aoid = tpls(i, 2).uint_value (), err);
        }      
      std::string name;
      if (! err)
        {
          SET_ERR (name = tpls(i, 1).string_value (), err);
        }
      std::string nspace;
      if (! err)
        {
          SET_ERR (nspace = tpls(i, 3).string_value (), err);
        }
      bool visible;
      if (! err)
        {
          SET_ERR (visible = tpls(i, 4).bool_value (), err);
        }
      if (err)
        {
          c_verror ("octave_pq_get_enum_types: could not read returned result");
          return 1;
        }

      // must be allocated and filled before creating the name map
      // entry, to get a remaining location for the c-string used as
      // key
      oct_pq_conv_t *t_conv = new oct_pq_conv_t;
      t_conv->oid = oid;
      t_conv->aoid = aoid;
      t_conv->el_oids = oct_pq_el_oids_t ();
      t_conv->conv_cache = oct_pq_conv_cache_t ();
      t_conv->is_composite = false;
      t_conv->is_enum = true;
      t_conv->is_not_constant = true;
      t_conv->name = nspace.append (".").append (name);
      t_conv->to_octave_str = &to_octave_str_text;
      t_conv->to_octave_bin = &to_octave_bin_text;
      t_conv->from_octave_str = &from_octave_str_text;
      t_conv->from_octave_bin = &from_octave_bin_text;

      // we trust there is always an array type in the table
      oct_pq_conv_t *&by_oid = conv_map[oid], *&by_aoid = conv_map[aoid],
        *&by_name = name_conv_map[t_conv->name.c_str ()];
      if (by_oid || by_aoid || by_name)
        {
          c_verror ("octave_pq_get_enum_types: internal error, key already in typemap");
          if (! by_oid) conv_map.erase (oid);
          if (! by_aoid) conv_map.erase (aoid);
          if (! by_name) name_conv_map.erase (t_conv->name.c_str ());
          delete t_conv;
          return 1;
        }

      by_oid = by_aoid = by_name = t_conv;

      if (visible)
        {
          oct_pq_conv_t *t_conv_v = new oct_pq_conv_t (*t_conv);

          t_conv_v->el_oids = oct_pq_el_oids_t ();

          t_conv_v->conv_cache = oct_pq_conv_cache_t ();

          t_conv_v->name = name;

          name_conv_map[t_conv_v->name.c_str ()] = t_conv_v;
        }
    }

  return 0;
}

int octave_pq_connection_rep::octave_pq_refresh_types (void)
{
  octave_pq_delete_non_constant_types ();

  // refresh unqualified base type names, may be replaced later with
  // user-defined types
  for (int i = 0; i < OCT_PQ_NUM_CONVERTERS; i++)
    name_conv_map[conv_ptrs[i]->name.c_str () + pq_bpl] = conv_ptrs[i];

  if (octave_pq_get_composite_types () || octave_pq_get_enum_types ())
    {
      PGconn *t_conn = conn;

      conn = NULL;

      if (t_conn)
        {
          BEGIN_INTERRUPT_IMMEDIATELY_IN_FOREIGN_CODE;
          PQfinish (t_conn);
          END_INTERRUPT_IMMEDIATELY_IN_FOREIGN_CODE;
        }

      c_verror ("octave_pq_refresh_types: could not read types");
      return 1;
    }
  else
    return 0;
}
