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

#ifndef __OCT_PQ_CONNECTION__

#define __OCT_PQ_CONNECTION__

#include <octave/oct.h>

#include <libpq-fe.h>

#include "converters.h"

class
octave_pq_connection_rep
{
public:

  octave_pq_connection_rep (std::string &);

  ~octave_pq_connection_rep (void);

  void octave_pq_close (void);

  int octave_pq_refresh_types (void);

  PGconn *octave_pq_get_conn (void) { return conn; }

  oct_pq_conv_map_t conv_map;

  oct_pq_name_conv_map_t name_conv_map;

  const bool &get_integer_datetimes (void) const { return integer_datetimes; }

private:

  PGconn *conn;

  // Oid of postgres_user, needed to distinguish base types from
  // others.
  octave_value postgres;

  void octave_pq_delete_non_constant_types (void);

  // returns zero on success
  int octave_pq_get_postgres_oid (void);

  // returns zero on success
  int octave_pq_fill_base_types (void);

  // returns zero on success
  int octave_pq_get_composite_types (void);

  // returns zero on success
  int octave_pq_get_enum_types (void);

  // server configuration
  bool integer_datetimes;
};

class
octave_pq_connection : public octave_base_value
{
public:

  octave_pq_connection (std::string &arg)
    : rep (new octave_pq_connection_rep (arg))
  {
    static bool type_registered = false;

    if (! type_registered)
      {
        register_type ();

        type_registered = true;
      }
  }

  ~octave_pq_connection (void) { delete rep; }

  octave_pq_connection_rep *get_rep (void) const { return rep; }


  // Octave internal stuff

  bool is_constant (void) const { return true; }

  bool is_defined (void) const { return true; }

  bool is_true (void) const { return rep->octave_pq_get_conn () != 0; }

  void print_raw (std::ostream& os, bool pr_as_read_syntax = false) const
  {
    indent (os);
    os << "<PGconn object>";
    newline (os);
  }

  void print (std::ostream& os, bool pr_as_read_syntax = false) const
  {
    print_raw (os);
  }
  // Octave changeset bcd71a2531d3 (Jan 31st 2014) made
  // octave_base_value::print() non-const, after that this virtual
  // function is not re-defined by the above print() function.  Having
  // both const and non-const print() here seems to work both with
  // Octave < and >= bcd71a2531d3 (print() is only called over the
  // parent class virtual function).
  void print (std::ostream& os, bool pr_as_read_syntax = false)
  {
    print_raw (os);
  }

  bool print_as_scalar (void) const { return true; }

private:

  // needed by Octave for register_type()
  octave_pq_connection (void) : rep (NULL) { }

  octave_pq_connection_rep *rep;

  DECLARE_OV_TYPEID_FUNCTIONS_AND_DATA
};

#endif // __OCT_PQ_CONNECTION__
