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
#include <octave/ov-struct.h>
#include <octave/Cell.h>
#include <octave/lo-ieee.h>

#include <iostream>
#include <fstream>

#include "command.h"
#include "converters.h"
#include "error-helpers.h"

#define COPY_HEADER_SIZE 19

#define COUT_RESIZE_STEP 1000 // resize result only after this number of rows

command::command (octave_pq_connection_rep &connection, std::string &cmd,
                  Cell &rtypes,std::string &who)
  : res (NULL), all_fetched (0), valid (1), conn (connection),
    rettypes (rtypes), caller (who)
{
  if (! (cptr = conn.octave_pq_get_conn ()))
    {
      valid = 0;
      c_verror ("%s: connection not open", caller.c_str ());
    }

  if (! PQsendQuery (cptr, cmd.c_str ()))
    {
      valid = 0;
      c_verror ("%s: could not dispatch command: %s", caller.c_str (),
                PQerrorMessage (cptr));
    }
  else
    {
      res = PQgetResult (cptr);
      check_first_result ();
    }
}

command::command (octave_pq_connection_rep &connection, std::string &cmd,
                  Cell &params, Cell &ptypes, Cell &rtypes, std::string &who)
  : res (NULL), all_fetched (1), valid (1), conn (connection),
    rettypes (rtypes), caller (who)
{
  if (! (cptr = conn.octave_pq_get_conn ()))
    {
      valid = 0;
      c_verror ("%s: connection not open", caller.c_str ());
    }

  int npars = params.numel ();

  char *vals [npars];
  std::vector<oct_pq_dynvec_t> valsvec;
  valsvec.resize (npars);
  int pl [npars];
  int pf [npars];
  Oid oids [npars];

  for (int i = 0; i < npars; i++)
    {
      pf[i] = 1; // means binary format

      if (params(i).is_real_scalar () && params(i).isna ().bool_value ())
        {
          vals[i] = NULL;

          oids[i] = 0;
        }
      else
        {
          pq_oct_type_t oct_type;
          oct_pq_conv_t *conv;

          if (ptypes(i).is_empty ())
            {
              oct_type = simple;

              if (! (conv = pgtype_from_octtype (conn, params(i))))
                {
                  valid = 0;
                  break;
                }

              // array not possible here
              oids[i] = conv->oid;
            }
          else
            {
              bool err;
              std::string s;
              SET_ERR (s = ptypes(i).string_value (), err);
              if (err)
                {
                  valid = 0;
                  c_verror ("%s: parameter type specification no string",
                            caller.c_str ());
                  break;
                }

              if (! (conv = pgtype_from_spec (conn, s, oct_type)))
                {
                  valid = 0;
                  break;
                }

              if (oct_type == array)
                oids[i] = conv->aoid;
              else
                oids[i] = conv->oid;
            }

          switch (oct_type)
            {
            case simple:
              if (conv->from_octave_bin (conn, params(i), valsvec[i]))
                valid = 0;
              break;

            case array:
              if (from_octave_bin_array (conn, params(i), valsvec[i], conv))
                valid = 0;
              break;

            case composite:
              if (from_octave_bin_composite (conn, params(i), valsvec[i], conv))
                valid = 0;
              break;

            default:
              // should not get here
              valid = 0;
              c_verror ("%s: internal error, undefined type identifier",
                        caller.c_str ());

            }

          if (! valid) break;

          vals[i] = &(valsvec[i].front ());
          pl[i] = valsvec[i].size ();
        }
    }

  if (valid)
    {
      BEGIN_INTERRUPT_IMMEDIATELY_IN_FOREIGN_CODE;

      res = PQexecParams (cptr, cmd.c_str (), npars, oids, vals, pl, pf, 1);

      END_INTERRUPT_IMMEDIATELY_IN_FOREIGN_CODE;

      check_first_result ();
    }
}

octave_map command::get_elements_typeinfo (oct_pq_conv_t *conv, bool &err)
{
  int nel = conv->el_oids.size ();

  octave_map ret (dim_vector (1, nel));
  Cell types_name (1, nel);
  Cell types_array (1, nel);
  Cell types_composite (1, nel);
  Cell types_enum (1, nel);
  Cell types_elements (1, nel);

  for (int i = 0; i < nel; i++)
    {
      oct_pq_conv_t *el_conv;
      pq_oct_type_t oct_type;

      if (! (el_conv = pgtype_from_spec (conn, conv->el_oids[i],
                                         conv->conv_cache[i], oct_type)))
        {
          err = true;
          return ret;
        }

      types_name(i) = octave_value (el_conv->name);
      types_array(i) = octave_value (oct_type == array);
      types_enum(i) = octave_value (el_conv->is_enum);
      types_composite(i) = octave_value (el_conv->is_composite);
      if (el_conv->is_composite)
        {
          bool rec_err = false;
          types_elements(i) = octave_value (get_elements_typeinfo (el_conv,
                                                                   rec_err));
          if (rec_err)
            {
              err = true;
              return ret;
            }
        }
    }

  ret.assign ("name", types_name);
  ret.assign ("is_array", types_array);
  ret.assign ("is_composite", types_composite);
  ret.assign ("is_enum", types_enum);
  ret.assign ("elements", types_elements);

  return ret;
}

octave_value command::process_single_result (const std::string &infile,
                                             const std::string &outfile,
                                             const Cell &cdata,
                                             const Cell &ctypes,
                                             bool coids,
                                             bool cin_var)
{
  octave_value retval;

  // first result is already fetched
  if (! res && (res = PQgetResult (cptr)))
    state = PQresultStatus (res);

  if (! res)
    all_fetched = 1;
  else
    {
      switch (state)
        {
        case PGRES_BAD_RESPONSE:
          valid = 0;
          c_verror ("%s: server response not understood", caller.c_str ());
          break;
        case PGRES_FATAL_ERROR:
          valid = 0;
          c_verror ("%s: fatal error: %s", caller.c_str (),
                    PQresultErrorMessage (res));
          break;
        case PGRES_COMMAND_OK:
          retval = command_ok_handler ();
          break;
        case PGRES_TUPLES_OK:
          retval = tuples_ok_handler ();
          break;
        case PGRES_COPY_OUT:
          retval = copy_out_handler (outfile);
          break;
        case PGRES_COPY_IN:
          retval = copy_in_handler (infile, cdata, ctypes, coids, cin_var);
          break;
        case PGRES_NONFATAL_ERROR:
          break;
        default:
          valid = 0;
          c_verror ("internal error, unexpected server response");
        }

      if (res) // could have been changed by a handler
        {
          PQclear (res);
          res = NULL;
        }
    }

  return retval;
}

octave_value command::tuples_ok_handler (void)
{
  octave_map ret;

  int nt = PQntuples (res);
  int nf = PQnfields (res);

  Cell data (nt, nf);
  Cell columns (1, nf);
  Cell types_name (1, nf);
  Cell types_array (1, nf);
  Cell types_composite (1, nf);
  Cell types_enum (1, nf);
  Cell types_elements (1, nf);
  octave_map types (dim_vector (1, nf));

  bool rtypes_given;
  int l = rettypes.numel ();
  if (l > 0)
    {
      if (l != nf)
        {
          valid = 0;
          c_verror ("%s: wrong number of given returned types",
                    caller.c_str ());
          return octave_value_list ();
        }
      rtypes_given = true;
    }
  else
    rtypes_given = false;

  for (int j = 0; j < nf; j++) // j is column
    {
      columns(j) = octave_value (PQfname (res, j));

      int f = PQfformat (res, j);

      oct_pq_to_octave_fp_t simple_type_to_octave;
      to_octave_array_fp_t array_to_octave;
      to_octave_composite_fp_t composite_to_octave;

      oct_pq_conv_t *conv = NULL; // silence inadequate warning by
                                  // initializing it here
      pq_oct_type_t oct_type;

      if (rtypes_given) // for internal reading of system tables
        {
          std::string type;
          bool err;
          SET_ERR (type = rettypes(j).string_value (), err);
          if (err)
            {
              valid = 0;
              c_verror ("%s: could not convert given type to string",
                        caller.c_str ());
              break;
            }
          else if (! (conv = pgtype_from_spec (conn, type, oct_type)))
            {
              valid = 0;
              break;
            }
        }
      else if (! (conv = pgtype_from_spec (conn, PQftype (res, j), oct_type)))
        {
          valid = 0;
          break;
        }

      if (f)
        {
          array_to_octave = &to_octave_bin_array;
          composite_to_octave = &to_octave_bin_composite;
          // will be NULL for non-simple converters
          simple_type_to_octave = conv->to_octave_bin;
        }
      else
        {
          array_to_octave = &to_octave_str_array;
          composite_to_octave = &to_octave_str_composite;
          // will be NULL for non-simple converters
          simple_type_to_octave = conv->to_octave_str;
        }

      // prepare type information
      types_name(j) = octave_value (conv->name);
      types_array(j) = octave_value (oct_type == array);
      types_enum(j) = octave_value (conv->is_enum);
      types_composite(j) = octave_value (conv->is_composite);
      if (conv->is_composite)
        {
          // To implement here: recursively go through the elements
          // and return respective recursive structures. This has the
          // side effect that all converters necessary for this query
          // will be looked up and cached (if they aren't already), so
          // in the actual conversion of composite types only cache
          // reads are performed, no map lookups.

          bool err = false;

          types_elements(j) = octave_value (get_elements_typeinfo (conv, err));

          if (err)
            {
              valid = 0;
              break;
            }
        }

      for (int i = 0; i < nt; i++) // i is row
        {
          if (PQgetisnull (res, i, j))
            data(i, j) = octave_value (octave_NA);
          else
            {
              char *v = PQgetvalue (res, i, j);
              int nb = PQgetlength (res, i, j);
              octave_value ov;

              switch (oct_type)
                {
                case simple:
                  if (simple_type_to_octave (conn, v, ov, nb))
                    valid = 0;
                  break;

                case array:
                  if (array_to_octave (conn, v, ov, nb, conv))
                    valid = 0;
                  break;

                case composite:
                  if (composite_to_octave (conn, v, ov, nb, conv))
                    valid = 0;
                  break;

                default:
                  // should not get here
                  c_verror ("%s: internal error, undefined type identifier",
                            caller.c_str ());

                  valid = 0;
                }

              if (valid)
                data(i, j) = ov;
              else
                break;
            }
        }

      if (! valid)
        break;
    }

  if (! valid)
    return octave_value_list ();
  else
    {
      ret.assign ("data", octave_value (data));
      ret.assign ("columns", octave_value (columns));

      types.setfield ("name", types_name);
      types.setfield ("is_array", types_array);
      types.setfield ("is_composite", types_composite);
      types.setfield ("is_enum", types_enum);
      types.setfield ("elements", types_elements);
      ret.assign ("types", octave_value (types));

      return octave_value (ret);
    }
}

octave_value command::copy_out_handler (const std::string &outfile)
{
  octave_value retval;

  if (! outfile.empty ())
    {
      // store unchecked output in file

      std::ofstream ostr (outfile.c_str (), std::ios_base::out);
      if (ostr.fail ())
        {
          valid = 0;
          c_verror ("could not open output file %s", outfile.c_str ());
          return retval;
        }

      char *data;
      int nb;
      while ((nb = PQgetCopyData (cptr, &data, 0)) > 0)
        {
          if (! (ostr.fail () || ostr.bad ()))
            {
              ostr.write (data, nb);
              if (ostr.bad ())
                c_verror ("write to file failed");
            }
          PQfreemem (data);
        }

      if (! ostr.bad ())
        ostr.close ();

      if (nb == -2)
        {
          valid = 0;
          c_verror ("server error in copy-out: %s", PQerrorMessage (cptr));
        }
      else
        {
          PQclear (res);

          if ((res = PQgetResult (cptr)))
            {
              if ((state = PQresultStatus (res)) == PGRES_FATAL_ERROR)
                {
                  valid = 0;
                  c_verror ("server error in copy-out: %s",
                            PQerrorMessage (cptr));
                }
            }
          else
            {
              valid = 0;
              c_verror ("unexpectedly got no result information");
            }
        }
    }
  else
    {
      valid = 0;
      c_verror ("no output file given");
    }

  return octave_value (std::string ("copy out"));
}

octave_value command::copy_in_handler (const std::string &infile,
                                       const Cell &data,
                                       const Cell &cin_types,
                                       bool oids,
                                       bool var)
{
  octave_value retval;

#define OCT_PQ_READSIZE 4096

  char buff [OCT_PQ_READSIZE];

  if (! var)
    {
      // read unchecked input from file

      if (infile.empty ())
        {
          valid = 0;

          c_verror ("no input file given");

          return retval;
        }

      std::ifstream istr (infile.c_str (), std::ios_base::in);
      if (istr.fail ())
        {
          c_verror ("could not open input file %s", infile.c_str ());

          PQputCopyEnd (cptr, "could not open input file");

          c_verror ("server error: %s", PQerrorMessage (cptr));

          valid = 0;

          return retval;
        }

      do
        {
          istr.read (buff, OCT_PQ_READSIZE);

          if (istr.bad ())
            {
              valid = 0;

              c_verror ("could not read file %s", infile.c_str ());

              break;
            }
          else
            {
              int nb;

              if ((nb = istr.gcount ()) > 0)
                if (PQputCopyData (cptr, buff, nb) == -1)
                  {
                    valid = 0;
                    
                    c_verror ("%s", PQerrorMessage (cptr));

                    break;
                  }
            }
        }
      while (! istr.eof ());

      istr.close ();

      if (! valid)
        {
          PQputCopyEnd (cptr, "copy-in interrupted");

          c_verror ("%s", PQerrorMessage (cptr));
        }
      else
        {
          if (PQputCopyEnd (cptr, NULL) == -1)
            {
              valid = 0;
              c_verror ("%s", PQerrorMessage (cptr));
            }
          else
            {
              PQclear (res);

              if ((res = PQgetResult (cptr)))
                {
                  if ((state = PQresultStatus (res)) == PGRES_FATAL_ERROR)
                    {
                      valid = 0;
                      c_verror ("server error in copy-in: %s",
                                PQerrorMessage (cptr));
                    }
                }
              else
                {
                  valid = 0;
                  c_verror ("unexpectedly got no result information");
                }
            }
        }
    }
  else
    {
      // copy in from octave variable

      dim_vector dv = data.dims ();
      octave_idx_type r = dv(0);
      octave_idx_type c = dv(1);

      octave_idx_type nf = PQnfields (res);
      if (c != nf + oids)
        {
          valid = 0;

          c_verror ("variable for copy-in has %i columns, but should have %i",
                    c, nf + oids);

          PQputCopyEnd
            (cptr, "variable for copy-in has wrong number of columns");
        }
      else if (! PQbinaryTuples (res))
        {
          valid = 0;

          c_verror ("copy-in from variable must use binary mode");

          PQputCopyEnd (cptr, "copy-in from variable must use binary mode");
        }
      else
        {
          for (octave_idx_type j = 0; j < nf; j++)
            if (! PQfformat (res, j))
              {
                valid = 0;

                c_verror ("copy-in from variable must use binary mode in all columns");

                PQputCopyEnd (cptr, "copy-in from variable must use binary mode in all columns");

                break;
              }
        }

      if (! valid)
        {
          c_verror ("server error: %s", PQerrorMessage (cptr));

          return retval;
        }

      char header [COPY_HEADER_SIZE];
      memset (header, 0, COPY_HEADER_SIZE);
      strcpy (header, "PGCOPY\n\377\r\n\0");
      uint32_t tpu32 = htobe32 (uint32_t (oids) << 16);
      memcpy (&header[11], &tpu32, 4);

      char trailer [2];
      int16_t tp16 = htobe16 (int16_t (-1));
      memcpy (&trailer, &tp16, 2);

      if (PQputCopyData (cptr, header, COPY_HEADER_SIZE) == -1)
        {
          PQputCopyEnd (cptr, "could not send header");

          valid = 0;

          c_verror ("server error: %s", PQerrorMessage (cptr));
        }
      else
        {
          oct_pq_conv_t *convs [c];
          memset (convs, 0, sizeof (convs));
          pq_oct_type_t oct_types [c];

          for (octave_idx_type i = 0; i < r; i++) // i is row
            {
              int16_t fc = htobe16 (int16_t (nf));
              if (PQputCopyData (cptr, (char *) &fc, 2) == -1)
                {
                  c_verror ("%s", PQerrorMessage (cptr));

                  PQputCopyEnd (cptr, "error sending field count");

                  c_verror ("server error: %s", PQerrorMessage (cptr));

                  valid = 0;

                  break;
                }

              // j is column of argument data
              for (octave_idx_type j = 0; j < c; j++)
                {
                  if (data(i, j).is_real_scalar () &&
                      data(i, j).isna ().bool_value ())
                    {
                      int32_t t = htobe32 (int32_t (-1));
                      if (PQputCopyData (cptr, (char *) &t, 4) == -1)
                        {
                          valid = 0;

                          c_verror ("could not send NULL in copy-in");

                          break;
                        }
                    }
                  else
                    {
                      if (! convs [j])
                        {
                          if ((j == 0) && oids)
                            {
                              std::string t ("oid");
                              if (! (convs[0] =
                                     pgtype_from_spec (conn, t, oct_types[0])))
                                {
                                  valid = 0;

                                  c_verror ("could not get converter for oid in copy-in");
                                  break;
                                }
                            }
                          else
                            {
                              if (cin_types(j).is_empty ())
                                {
                                  oct_types[j] = simple;

                                  if (! (convs[j] =
                                         pgtype_from_octtype (conn,
                                                              data(i, j))))
                                    {
                                      valid = 0;

                                      c_verror ("could not determine type in column %i for copy-in",
                                                j);

                                      break;
                                    }
                                }
                              else
                                {
                                  bool err;
                                  std::string s;
                                  SET_ERR (s = cin_types(j).string_value (),
                                           err);
                                  if (err)
                                    {
                                      valid = 0;

                                      c_verror ("column type specification no string");

                                      break;
                                    }

                                  if (! (convs[j] =
                                         pgtype_from_spec (conn, s,
                                                           oct_types[j])))
                                    {
                                      valid = 0;

                                      c_verror ("invalid column type specification");

                                      break;
                                    }
                                }
                            }
                        } // ! convs [j]

                      oct_pq_dynvec_t val;

                      bool conversion_failed = false;
                      switch (oct_types[j])
                        {
                        case simple:
                          if (convs[j]->from_octave_bin (conn, data(i, j), val))
                            conversion_failed = true;
                          break;

                        case array:
                          if (from_octave_bin_array (conn, data(i, j), val,
                                                     convs[j]))
                            conversion_failed = true;
                          break;

                        case composite:
                          if (from_octave_bin_composite (conn, data(i, j), val,
                                                         convs[j]))
                            conversion_failed = true;
                          break;

                        default:
                          // should not get here
                          c_verror ("internal error, undefined type identifier");
                          conversion_failed = true;
                        }

                      if (conversion_failed)
                        {
                          valid = 0;
                          error ("could not convert data(%i, %i) for copy-in",
                                 i, j);
                        }
                      else
                        {
                          uint32_t t = htobe32 (uint32_t (val.size ()));
                          if (PQputCopyData (cptr, (char *) &t, 4) == -1)
                            {
                              valid = 0;
                              c_verror ("could not send data length in copy-in");
                            }
                          else if (PQputCopyData (cptr, &(val.front ()),
                                                  val.size ()) == -1)
                            {
                              valid = 0;
                              c_verror ("could not send copy-in data");
                            }
                        }

                      if (! valid) break;
                    }
                } // columns of argument data

              if (! valid)
                {
                  PQputCopyEnd (cptr, "error sending copy-in data");

                  c_verror ("server error: %s", PQerrorMessage (cptr));

                  break;
                }
            } // rows of argument data
        }

      if (valid)
        if (PQputCopyData (cptr, trailer, 2) == -1)
          {
            valid = 0;

            PQputCopyEnd (cptr, "could not send trailer");

            c_verror ("%s", PQerrorMessage (cptr));
          }

      if (valid)
        {
          if (PQputCopyEnd (cptr, NULL) == -1)
            {
              valid = 0;
              c_verror ("%s", PQerrorMessage (cptr));
            }
          else
            {
              PQclear (res);

              if ((res = PQgetResult (cptr)))
                {
                  if ((state = PQresultStatus (res)) == PGRES_FATAL_ERROR)
                    {
                      valid = 0;
                      c_verror ("server error in copy-in: %s",
                                PQerrorMessage (cptr));
                    }
                }
              else
                {
                  valid = 0;
                  c_verror ("unexpectedly got no result information");
                }
            }
        }
    } // copy from variable

  return octave_value (std::string ("copy in"));
}
