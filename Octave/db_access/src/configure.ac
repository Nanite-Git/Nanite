#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.
#
### Copyright (C) 2015, 2016 Olaf Till <i7tiol@t-online.de>
###
### This program is free software; you can redistribute it and/or
### modify it under the terms of the GNU General Public License as
### published by the Free Software Foundation; either version 3 of the
### License, or (at your option) any later version.
###
### This program is distributed in the hope that it will be useful,
### but WITHOUT ANY WARRANTY; without even the implied warranty of
### MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
### General Public License for more details.
###
### You should have received a copy of the GNU General Public License
### along with this program; if not, see
### <http://www.gnu.org/licenses/>.

AC_PREREQ([2.67])
AC_INIT([database], [2.4.2], [i7tiol@t-online.de])
AC_CONFIG_SRCDIR([__pq_connect__.cc])
AC_CONFIG_HEADERS([config.h])

# Checks for programs.
AC_CHECK_PROG(MKOCTFILE, mkoctfile, mkoctfile)
if test -z "$MKOCTFILE"; then
AC_MSG_ERROR([mkoctfile not found], 1);
fi
AC_CHECK_PROG(OCTAVE_CONFIG, octave-config, octave-config)
if test -z "$OCTAVE_CONFIG"; then
AC_MSG_ERROR([octave-config not found], 1);
fi
AC_CHECK_PROG(PG_CONFIG, pg_config, pg_config)
if test -z "$PG_CONFIG"; then
AC_MSG_ERROR([pg_config not found], 1);
fi
AC_PROG_AWK
if test -z "$ac_cv_prog_AWK"; then
AC_MSG_ERROR([AWK not found], 1);
fi

AC_PROG_SED

# The same value of CXX as Octave was compiled with is supposed to be used.
CXX=${CXX:-`${MKOCTFILE} -p CXX`}
AC_PROG_CXX
  
AC_PROG_CXXCPP

# Checks for libraries.
AC_CHECK_LIB([pq], [PQconnectdb], [], [
  AC_MSG_ERROR([unable to find the PQconnectdb() function in pq])
])

# Checks for typedefs, structures, and compiler characteristics.

# Checks for library functions.
AC_FUNC_ERROR_AT_LINE

# Check for PostgreSQL version.
AC_MSG_CHECKING([whether PostgreSQL version >= 8.3])
if `$PG_CONFIG --version | awk 'NR==1 { print $(NF); };' | awk -F '.' ' { if ( $1 > 8 || $1 == 8 && $2 >= 3 ) exit 1; else exit 0}'`; then
  AC_MSG_RESULT(no)
  AC_MSG_ERROR([package \"database\" does not work with PostgreSQL version < 8.3], 1);
else
  AC_MSG_RESULT(yes)
fi

# Start of checks for Octave features, preparations for checks.
OCTLIBDIR=${OCTLIBDIR:-`$OCTAVE_CONFIG -p OCTLIBDIR`}
## We need Octaves include path both with and without '/octave'
## appended. The path without '/octave' is needed to selectively test
## for Octave headers, like octave/....h. The path with '/octave' is
## needed since some Octave headers contain include directives for
## other Octave headers with <> instead of "".
OCTINCLUDEDIR=${OCTINCLUDEDIR:-`$OCTAVE_CONFIG -p OCTINCLUDEDIR`}
TPINCLUDEDIR=`echo $OCTINCLUDEDIR | $SED -e 's/\/octave$//'`
AC_LANG_PUSH([C++])
TCXXFLAGS=$CXXFLAGS
TLDFLAGS=$LDFLAGS
TLIBS=$LIBS
TCPPFLAGS=$CPPFLAGS
LDFLAGS="-L$OCTLIBDIR $LDFLAGS"
LIBS="-loctinterp $LIBS"
# CXXFLAGS=
CPPFLAGS="-I$OCTINCLUDEDIR -I$TPINCLUDEDIR $CPPFLAGS"

## Presence of 'error_state' -- does _not_ indicate no exceptions are
## used.
AC_LINK_IFELSE(
  [AC_LANG_PROGRAM([[#include <octave/oct.h>]
                    [#include <stdio.h>]],
                   [[printf ("%i", error_state);]])],
  [AC_DEFINE([HAVE_OCTAVE_ERROR_STATE], 1)])

## Presence of 'verror (octave_execution_exception&, const char *,
## va_list)'
AC_LINK_IFELSE(
  [AC_LANG_PROGRAM([[#include <octave/oct.h>]],
                   [[octave_execution_exception e;]
                    [va_list args;]
                    [verror (e, "test", args);]])],
  [AC_DEFINE([HAVE_OCTAVE_VERROR_ARG_EXC], 1)])

LIBS=$TLIBS
LDFLAGS=$TLDFLAGS
CXXFLAGS=$TCXXFLAGS
CPPFLAGS=$TCPPFLAGS
AC_LANG_POP([C++])
# End of checks for Octave features.

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
