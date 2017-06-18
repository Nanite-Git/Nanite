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

#include "command.h"

// PKG_ADD: autoload ("pq_update_types", "pq_interface.oct");
// PKG_DEL: autoload ("pq_update_types", "pq_interface.oct", "remove");

DEFUN_DLD (pq_update_types, args, ,
           "-*- texinfo -*-\n\
@deftypefn {Loadable Function} pq_update_types (@var{connection})\n\
Updates information on existing postgresql types for @var{connection}. Use this before @code{pq_exec_params} if types were created or dropped while the connection was already established or if the schema search path changed. A newly created connection will automatically retrieve this information at connection time.\n\
\n\
@end deftypefn")
{
  std::string fname ("pq_update_types");

  octave_value_list retval;

  if (args.length () != 1 ||
      args(0).type_id () != octave_pq_connection::static_type_id ())
    {
      print_usage ();

      return retval;
    }

  const octave_base_value& rep = (args(0).get_rep ());

  const octave_pq_connection &oct_pq_conn =
    dynamic_cast<const octave_pq_connection&> (rep);

  if (oct_pq_conn.get_rep ()->octave_pq_refresh_types ())
    error ("%s failed", fname.c_str ());

  return retval;
}
