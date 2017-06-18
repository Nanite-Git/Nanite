/*

Copyright (C) 2016 Olaf Till <i7tiol@t-online.de>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; If not, see <http://www.gnu.org/licenses/>.

*/

#include "config.h"

// call verror
#ifdef HAVE_OCTAVE_VERROR_ARG_EXC
void c_verror (octave_execution_exception&, const char *, ...);
#else
void c_verror (const octave_execution_exception&, const char *, ...);
#endif

// call verror
void c_verror (const char *fmt, ...);

// Print a message if 'code' causes an error and raise an error again,
// both if Octave uses exceptions for errors and if it still uses
// error_state. In the latter case return 'retval'.
#ifdef HAVE_OCTAVE_ERROR_STATE
  // can throw octave_execution_exception despite of this
  #define CHECK_ERROR(code, retval, ...)      \
    try \
      { \
        code ; \
 \
        if (error_state) \
          { \
            error (__VA_ARGS__); \
 \
            return retval; \
          } \
      } \
    catch (octave_execution_exception& e) \
      { \
        c_verror (e, __VA_ARGS__); \
 \
        throw e; \
      }
#else
  #define CHECK_ERROR(code, retval, ...) \
    try \
      { \
        code ; \
      } \
    catch (octave_execution_exception& e) \
      { \
        c_verror (e, __VA_ARGS__); \
 \
        throw e; \
      }
#endif

// If 'code' causes an error, print a message and call exit(1) if
// Octave doesn't throw exceptions for errors but still uses
// error_state.
#ifdef HAVE_OCTAVE_ERROR_STATE
  // can throw octave_execution_exception despite of this
  #define CHECK_ERROR_EXIT1(code, ...) \
    try \
      { \
        code ; \
 \
        if (error_state) \
          { \
            c_verror (__VA_ARGS__); \
 \
            exit (1); \
          } \
      } \
    catch (octave_execution_exception& e) \
      { \
        c_verror (e, __VA_ARGS__); \
 \
        exit (1); \
      }
#else
  #define CHECK_ERROR_EXIT1(code, ...) \
    try \
      { \
        code ; \
      } \
    catch (octave_execution_exception& e) \
      { \
        c_verror (e, __VA_ARGS__); \
 \
        exit (1); \
      }
#endif

// Set 'err' to true if 'code' causes an error, else to false; both if
// Octave uses exceptions for errors and if it still uses
// error_state. In the latter case reset error_state to 0.
#ifdef HAVE_OCTAVE_ERROR_STATE
  // can throw octave_execution_exception despite of this
  #define SET_ERR(code, err) \
    err = false; \
 \
    try \
      { \
        code ; \
        if (error_state) \
          { \
            error_state = 0; \
            err = true; \
          } \
      } \
    catch (octave_execution_exception&) \
      { \
        err = true; \
      }
#else
  #define SET_ERR(code, err) \
    try \
      { \
        code ; \
      } \
    catch (octave_execution_exception&) \
      { \
        err = true; \
      }
#endif
