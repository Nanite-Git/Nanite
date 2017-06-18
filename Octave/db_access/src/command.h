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

#ifndef __OCT_PQ_COMMAND__

#define __OCT_PQ_COMMAND__

#include <octave/oct.h>
#include <octave/ov-struct.h>
#include <octave/Cell.h>

#include "pq_connection.h"
#include "error-helpers.h"

class
command
{
public:

  command (octave_pq_connection_rep &connection, std::string &cmd, Cell &rtypes,
           std::string &who);

  command (octave_pq_connection_rep &connection, std::string &cmd, Cell &params,
           Cell &ptypes, Cell &rtypes, std::string &who);

  ~command (void)
  {
    if (res) PQclear (res);

    if (! all_fetched)
      {
        while ((res = PQgetResult (cptr)))
          PQclear (res);
      }
  }

  typedef int (*to_octave_array_fp_t)
    (const octave_pq_connection_rep &, const char *, octave_value &,
     int, oct_pq_conv_t *);

  typedef int (*to_octave_composite_fp_t)
    (const octave_pq_connection_rep &, const char *, octave_value &,
     int, oct_pq_conv_t *);

  int all_results_fetched (void)
  {
    return all_fetched;
  }

  octave_value process_single_result (void)
  {
    Cell c;
    // inlining should prevent the additional copy
    return process_single_result ("", "", c, c, false, false);
  }

  octave_value process_single_result (const std::string &infile,
                                      const std::string &outfile,
                                      const Cell &cin_data,
                                      const Cell &cin_types,
                                      bool cin_with_oids,
                                      bool cin_from_variable);

  int good (void) {return valid;}

private:

  octave_map get_elements_typeinfo (oct_pq_conv_t *conv, bool &err);

  void check_first_result (void)
    {
      if (! res)
        {
          valid = 0;
          c_verror ("%s: could not execute command: %s", caller.c_str (),
                    PQerrorMessage (cptr));

        }
      else if ((state = PQresultStatus (res)) == PGRES_EMPTY_QUERY)
        {
          valid = 0;
          c_verror ("%s: empty command", caller.c_str ());
        }
    }

  octave_value command_ok_handler (void)
  {
    char *c = PQcmdTuples (res);

    return octave_value (atoi (c));
  }

  octave_value tuples_ok_handler (void);

  octave_value copy_out_handler (const std::string &);

  octave_value copy_in_handler (const std::string &, const Cell &, const Cell &,
                                bool, bool);

  PGresult *res;
  int all_fetched;
  int valid;
  ExecStatusType state;
  octave_pq_connection_rep &conn;
  PGconn *cptr;
  Cell &rettypes;
  std::string &caller;

};

#endif // __OCT_PQ_COMMAND__
