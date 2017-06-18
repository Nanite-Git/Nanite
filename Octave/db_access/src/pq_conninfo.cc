/*

Copyright (C) 2013-2016 Olaf Till <i7tiol@t-online.de>

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

#include "pq_connection.h"
#include "error-helpers.h"

// PKG_ADD: autoload ("pq_conninfo", "pq_interface.oct");
// PKG_DEL: autoload ("pq_conninfo", "pq_interface.oct", "remove");

DEFUN_DLD (pq_conninfo, args, ,
           "-*- texinfo -*-\n\
@deftypefn {Loadable Function} {@var{val} =} pq_conninfo (@var{connection}, @var{label})\n\
Retrieves connection information for postgresql connection @var{connection}, specified by the string @var{label}, and returns the value of this information in @var{val}. The type of @var{val} depends on the requested information. Currently, the only recognized @var{label} is @code{'integer_datetimes'}; @var{val} is @code{true} if 8-byte date and time values are stored as integer in the server, and @code{false} if they are stored as @code{double} (which is deprecated).\n\
@end deftypefn")
{
  std::string fname ("pq_conninfo");

  octave_value retval;

  if (args.length () != 2 ||
      args(0).type_id () != octave_pq_connection::static_type_id ())
    {
      print_usage ();

      return retval;
    }

  std::string label;
  CHECK_ERROR (label = args(1).string_value (), retval,
               "%s: second argument can not be converted to a string",
               fname.c_str ());

  if (label.compare ("integer_datetimes"))
    {
      error ("%s: unrecognized label %s", fname.c_str (), label.c_str ());

      return retval;
    }

  const octave_base_value& rep = (args(0).get_rep ());

  const octave_pq_connection &oct_pq_conn =
    dynamic_cast<const octave_pq_connection&> (rep);

  PGconn *conn = oct_pq_conn.get_rep ()->octave_pq_get_conn ();

  if (! conn)
    {
      error ("%s: connection not open", fname.c_str ());

      return retval;
    }

  return octave_value (oct_pq_conn.get_rep ()->get_integer_datetimes ());
}
