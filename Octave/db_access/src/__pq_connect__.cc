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

#include "pq_connection.h"
#include "error-helpers.h"


// PKG_ADD: autoload ("__pq_connect__", "pq_interface.oct");
// PKG_DEL: autoload ("__pq_connect__", "pq_interface.oct", "remove");

DEFUN_DLD (__pq_connect__, args, ,
           "-*- texinfo -*-\n\
@deftypefn {Loadable Function} {@var{id}} __pq_connect__ (@var{options})\n\
Text.\n\
\n\
@end deftypefn")
{
  std::string fname ("__pq_connect__");

  octave_value retval;

  if (args.length () != 1)
    {
      print_usage ();

      return octave_value_list ();
    }

  std::string opt_string;
  CHECK_ERROR (opt_string = args(0).string_value (), retval,
               "%s: argument not a string", fname.c_str ());

  octave_pq_connection *oct_pq_conn = new octave_pq_connection (opt_string);

  if (! oct_pq_conn->get_rep ()->octave_pq_get_conn ())
    error ("%s failed", fname.c_str ());
  else
    retval = oct_pq_conn;

  return retval;
}
