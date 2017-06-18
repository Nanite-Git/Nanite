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

#include <octave/ls-mat4.h>

DEFUN_DLD (var2bytea, args, nout,
           "-*- texinfo -*-\n\
@deftypefn {Loadable Function} {} var2bytea (@var{value}, @dots{})\n\
Save input values in uint8 arrays in Octaves binary save format.\n\
\n\
This function returns as many output variables as input variables\n\
are given.\n\
\n\
The function can be used to prepare storage of Octave variable\n\
values as binary strings in a database, if the variables types have\n\
no corresponding SQL type.\n\
Each variable type which can be correctly saved and loaded with\n\
Octaves @code{save} and @code{load} functions is acceptable.\n\
The variable value can be restored with the function\n\
@code{bytea2var}.\n\
\n\
Note that the inputs are not variable names, but values.\n\
No variable names are saved.\n\
Loading the uint8 array with @code{load} (possibly after dumping\n\
it to a file) will not work.\n\
\n\
Example: to convert the first and third column of a cell-array @code{c},\n\
@code{[c@{:, [1, 3]@}] = var2bytea (c@{:, [1, 3]@});}\n\
can be used.\n\
\n\
@seealso{bytea2var}\n\
@end deftypefn")
{
  std::string fname ("var2bytea");

  octave_idx_type nargs = args.length ();

  octave_idx_type nvars = nout < nargs ? nout : nargs;

  octave_value_list retval (nvars);

  for (octave_idx_type i = 0; i < nvars; i++)
    {
      std::ostringstream os;

      /*
        slightly changed from load-save.cc (write_header(,LS_BINARY))
        to reduce storage size
      */

      os << (oct_mach_info::words_big_endian () ? "1B" : "1L");

      oct_mach_info::float_format flt_fmt =
        oct_mach_info::native_float_format ();

      char tmp = static_cast<char> (float_format_to_mopt_digit (flt_fmt));

      os.write (&tmp, 1);

      /*
        Much here is cut-and-pasted from ls-oct-binary.cc
        (save_binary_data()) in Octave.
      */

      // Write the string corresponding to the octave_value type.
      std::string typ = args(i).type_name ();
      int32_t len = typ.length ();
      os.write (reinterpret_cast<char *> (&len), 4);
      const char *btmp = typ.data ();
      os.write (btmp, len);

      // The octave_value of args(i) is const. Make a copy...
      octave_value val = args(i);
  
      // Call specific save function
      bool save_as_floats = false;
      if (! val.save_binary (os, save_as_floats) || ! os)
        {
          error ("%s: could not save variable %i", fname.c_str (), i + 1);

          return retval;
        }

      std::string s (os.str ());

      uint8NDArray m (dim_vector (s.length (), 1));

      memcpy (m.fortran_vec (), s.data (), s.length ());

      retval(i) = octave_value (m);
    }

  return retval;
}

