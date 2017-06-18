/*

Copyright (C) 2016 Olaf Till <i7tiol@t-online.de>

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

#include <octave/load-save.h>

#include <octave/byte-swap.h>

#include <octave/ls-mat4.h>

#include "error-helpers.h"

DEFUN_DLD (bytea2var, args, nout,
           "-*- texinfo -*-\n\
@deftypefn {Loadable Function} {} bytea2var (@var{value}, @dots{})\n\
Restore variable values from uint8 arrays generated with @code{var2bytea}.\n\
\n\
Returns as many output variables as input variables are given.\n\
\n\
@seealso{var2bytea}\n\
@end deftypefn")
{
  std::string fname ("bytea2var");

  octave_idx_type nargs = args.length ();

  octave_idx_type nvars = nout < nargs ? nout : nargs;

  octave_value_list retval (nvars);

  bool err = false;

  octave_idx_type i;

  for (i = 0; i < nvars; i++)
    {
      uint8NDArray m;

      octave_value val;

      CHECK_ERROR (m = args(i).uint8_array_value (), retval,
                   "%s: could not convert argument %i to uint8 array",
                   fname.c_str (), i + 1);

      octave_idx_type nel = m.numel ();

      std::string s ((char *) m.fortran_vec (), nel);

      std::istringstream is (s);

      bool swap;
      oct_mach_info::float_format flt_fmt;

      /*
        slightly changed from load-save.cc (read_binary_file_header())
        to reduce storage size
      */

      const int magic_len = 2;
      char magic[magic_len+1];
      is.read (magic, magic_len);
      magic[magic_len] = '\0';

      if (strncmp (magic, "1L", magic_len) == 0)
        swap = oct_mach_info::words_big_endian ();
      else if (strncmp (magic, "1B", magic_len) == 0)
        swap = ! oct_mach_info::words_big_endian ();
      else
        {
          error ("%s: could not read binary header", fname.c_str ());

          return retval;
        }

      char tmp = 0;
      is.read (&tmp, 1);

      flt_fmt = mopt_digit_to_float_format (tmp);

      if (flt_fmt == oct_mach_info::flt_fmt_unknown)
        {
          error ("%s: unrecognized binary format", fname.c_str ());

          return retval;
        }

      int32_t len;
      if (! is.read (reinterpret_cast<char *> (&len), 4))
        {
          err = true;

          break;
        }

      if (swap)
        swap_bytes<4> (&len);

      {
        OCTAVE_LOCAL_BUFFER (char, buf, len+1);
        if (! is.read (buf, len))
          {
            err = true;

            break;
          }
        buf[len] = '\0';
        std::string typ (buf);
        val = octave_value_typeinfo::lookup_type (typ);
      }

      if (! val.load_binary (is, swap, flt_fmt))
        {
          err = true;

          break;
        }

      retval(i) = val;
    }

  if (err)
    error ("%s: could not load variable %i", fname.c_str (), i + 1);

  return retval;
}
