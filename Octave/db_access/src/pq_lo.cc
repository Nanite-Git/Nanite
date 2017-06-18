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
#include <octave/ov-struct.h>
#include <octave/Cell.h>

#include <stdio.h>

#include "command.h"
#include "error-helpers.h"
#include <libpq/libpq-fs.h>

// PKG_ADD: autoload ("pq_lo_import", "pq_interface.oct");
// PKG_ADD: autoload ("pq_lo_export", "pq_interface.oct");
// PKG_ADD: autoload ("pq_lo_unlink", "pq_interface.oct");
// PKG_DEL: autoload ("pq_lo_import", "pq_interface.oct", "remove");
// PKG_DEL: autoload ("pq_lo_export", "pq_interface.oct", "remove");
// PKG_DEL: autoload ("pq_lo_unlink", "pq_interface.oct", "remove");

#define OCT_PQ_BUFSIZE 1024

// For cleanup handling this is a class.
class pipe_to_lo
{
public:

  pipe_to_lo (octave_pq_connection_rep &, const char *, bool, std::string &);

  ~pipe_to_lo (void);

  bool valid (void) { return oid_valid; }

  Oid get_oid (void) { return oid; }

  std::string &msg;

private:

  octave_pq_connection_rep &oct_pq_conn;

  PGconn *conn;

  Oid oid;

  FILE *fp;

  bool oid_valid;

  int lod;

  bool commit;
};

pipe_to_lo::pipe_to_lo (octave_pq_connection_rep &a_oct_pq_conn,
                        const char *cmd, bool acommit, std::string &amsg)
  : msg (amsg), oct_pq_conn (a_oct_pq_conn),
    conn (a_oct_pq_conn.octave_pq_get_conn ()), oid (0), fp (NULL),
    oid_valid (false), lod (-1), commit (acommit)
{
  BEGIN_INTERRUPT_IMMEDIATELY_IN_FOREIGN_CODE;
  oid = lo_creat (conn, INV_WRITE);
  END_INTERRUPT_IMMEDIATELY_IN_FOREIGN_CODE;
  if (! oid || oid == InvalidOid)
    {
      msg = PQerrorMessage (conn);

      oid = 0;

      return;
    }

  if (! (fp = popen (cmd, "r")))
    {
      msg = "could not create pipe";

      return;
    }

  BEGIN_INTERRUPT_IMMEDIATELY_IN_FOREIGN_CODE;
  lod = lo_open (conn, oid, INV_WRITE);
  END_INTERRUPT_IMMEDIATELY_IN_FOREIGN_CODE;
  if (lod == -1)
    {
      msg = PQerrorMessage (conn);

      return;
    }

  char buff [OCT_PQ_BUFSIZE];

  int nb = 0, pnb = 0; // silence inadequate warnings by initializing
                       // them

  while (true)
    {
      BEGIN_INTERRUPT_IMMEDIATELY_IN_FOREIGN_CODE;
      nb = fread (buff, 1, OCT_PQ_BUFSIZE, fp);
      END_INTERRUPT_IMMEDIATELY_IN_FOREIGN_CODE;

      if (! nb) break;

      BEGIN_INTERRUPT_IMMEDIATELY_IN_FOREIGN_CODE;
      pnb = lo_write (conn, lod, buff, nb);
      END_INTERRUPT_IMMEDIATELY_IN_FOREIGN_CODE;
      if (pnb != nb)
        {
          msg = PQerrorMessage (conn);

          break;
        }
      }
  if (nb) return;

  if (pclose (fp) == -1)
    c_verror ("error closing pipe");

  fp = NULL;

  if (lo_close (conn, lod))
    msg = PQerrorMessage (conn);
  else
    oid_valid = true;

  lod = -1;
}

pipe_to_lo::~pipe_to_lo (void)
{
  if (lod != -1)
    {
      if (lo_close (conn, lod))
        c_verror ("%s", PQerrorMessage (conn));

      lod = -1;
    }

  if (oid && ! oid_valid)
    {
      if (lo_unlink (conn, oid) == -1)
        c_verror ("error unlinking new large object with oid %i", oid);
    }
  else
    oid = 0;

  if (fp)
    {
      if (pclose (fp) == -1)
        c_verror ("error closing pipe");

      fp = NULL;
    }

  if (commit)
    {
      std::string cmd ("commit;");
      Cell params;
      Cell ptypes (1, 0);
      Cell rtypes;
      std::string caller ("pq_lo_import");
      command c (oct_pq_conn, cmd, params, ptypes, rtypes, caller);

      if (c.good ())
        c.process_single_result ();

      if (! c.good ())
        c_verror ("%s: could not commit", caller.c_str ());
    }
}

// For cleanup handling this is a class.
class lo_to_pipe
{
public:

  lo_to_pipe (octave_pq_connection_rep &, Oid, const char *, bool, std::string &);

  ~lo_to_pipe (void);

  bool valid (void) { return success; }

  std::string &msg;

private:

  octave_pq_connection_rep &oct_pq_conn;

  PGconn *conn;

  Oid oid;

  FILE *fp;

  bool success;

  int lod;

  bool commit;
};

lo_to_pipe::lo_to_pipe (octave_pq_connection_rep &a_oct_pq_conn, Oid aoid,
                        const char *cmd, bool acommit, std::string &amsg) :
  msg (amsg), oct_pq_conn (a_oct_pq_conn),
  conn (a_oct_pq_conn.octave_pq_get_conn ()), oid (aoid), fp (NULL),
  success (false), lod (-1), commit (acommit)
{
  if (! (fp = popen (cmd, "w")))
    {
      msg = "could not create pipe";

      return;
    }

  BEGIN_INTERRUPT_IMMEDIATELY_IN_FOREIGN_CODE;
  lod = lo_open (conn, oid, INV_READ);
  END_INTERRUPT_IMMEDIATELY_IN_FOREIGN_CODE;
  if (lod == -1)
    {
      msg = PQerrorMessage (conn);

      return;
    }

  char buff [OCT_PQ_BUFSIZE];

  int nb = 0, pnb = 0; // silence inadequate warnings by initializing
                       // them

  while (true)
    {
      BEGIN_INTERRUPT_IMMEDIATELY_IN_FOREIGN_CODE;
      pnb = lo_read (conn, lod, buff, OCT_PQ_BUFSIZE);
      END_INTERRUPT_IMMEDIATELY_IN_FOREIGN_CODE;

      if (pnb == -1)
        {
          msg = PQerrorMessage (conn);

          break;
        }

      if (! pnb) break;

      BEGIN_INTERRUPT_IMMEDIATELY_IN_FOREIGN_CODE;
      nb = fwrite (buff, 1, pnb, fp);
      END_INTERRUPT_IMMEDIATELY_IN_FOREIGN_CODE;
      if (nb != pnb)
        {
          msg = "error writing to pipe";

          break;
        }
    }
  if (pnb) return;

  if (pclose (fp) == -1)
    c_verror ("error closing pipe");

  fp = NULL;

  if (lo_close (conn, lod))
    msg = PQerrorMessage (conn);
  else
    success = true;

  lod = -1;
}

lo_to_pipe::~lo_to_pipe (void)
{
  if (lod != -1)
    {
      if (lo_close (conn, lod))
        c_verror ("%s", PQerrorMessage (conn));

      lod = -1;
    }

  if (fp)
    {
      if (pclose (fp) == -1)
        c_verror ("error closing pipe");

      fp = NULL;
    }

  if (commit)
    {
      std::string cmd ("commit;");
      Cell params;
      Cell ptypes (1, 0);
      Cell rtypes;
      std::string caller ("pq_lo_export");
      command c (oct_pq_conn, cmd, params, ptypes, rtypes, caller);

      if (c.good ())
        c.process_single_result ();

      if (! c.good ())
        c_verror ("%s: could not commit", caller.c_str ());
    }
}

DEFUN_DLD (pq_lo_import, args, ,
           "-*- texinfo -*-\n\
@deftypefn {Loadable Function} {@var{oid} =} pq_lo_import (@var{connection}, @var{path})\n\
Imports the file in @var{path} on the client side as a large object into the database associated with @var{connection} and returns the Oid of the new large object. If @var{path} ends with a @code{|}, it is take as a shell command whose output is piped into a large object.\n\
@end deftypefn")
{
  std::string fname ("pq_lo_import");

  octave_value retval;

  if (args.length () != 2 ||
      args(0).type_id () != octave_pq_connection::static_type_id ())
    {
      print_usage ();

      return retval;
    }

  std::string path;
  CHECK_ERROR (path = args(1).string_value (), retval,
               "%s: second argument can not be converted to a string",
               fname.c_str ());

  bool from_pipe = false;
  unsigned int l = path.size ();
  if (l && path[l - 1] == '|')
    {
      unsigned int pos;
      // There seemed to be a bug in my C++ library so that
      // path.find_last_not_of (" \t\n\r\f", l - 1))
      // returned l - 1 ! This is the workaround.
      path.erase (l - 1, 1);
      if ((pos = path.find_last_not_of (" \t\n\r\f"))
          == std::string::npos)
        {
          error ("%s: no command found to pipe from", fname.c_str ());

          return retval;
        }
      path.erase (pos + 1, std::string::npos);

      from_pipe = true;
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

  bool make_tblock = false;
  switch (PQtransactionStatus (conn))
    {
    case PQTRANS_IDLE:
      make_tblock = true;
      break;
    case PQTRANS_INTRANS:
      break;
    case PQTRANS_INERROR:
      error ("%s: can't manipulate large objects within a failed transaction block",
             fname.c_str ());
      return retval;
    case PQTRANS_UNKNOWN:
      error ("%s: connection is bad", fname.c_str ());
      return retval;
    default: // includes PQTRANS_ACTIVE
      error ("%s: unexpected connection state", fname.c_str ());
      return retval;
    }

  if (make_tblock)
    {
      std::string cmd ("begin;");
      Cell params;
      Cell ptypes (1, 0);
      Cell rtypes;
      command c (*(oct_pq_conn.get_rep ()), cmd, params, ptypes, rtypes, fname);

      if (c.good ())
        c.process_single_result ();

      if (! c.good ())
        {
          error ("%s: could not begin transaction", fname.c_str ());
          return retval;
        }
    }

  Oid oid = 0;

  bool import_error = false;
  std::string msg;

  if (from_pipe)
    {
      pipe_to_lo tp (*(oct_pq_conn.get_rep ()), path.c_str (), make_tblock, msg);

      make_tblock = false; // commit handled by destructor of pipe_to_lo

      if (tp.valid ())
        oid = tp.get_oid ();
      else
        import_error = true;
    }
  else
    if (! (oid = lo_import (conn, path.c_str ())))
      {
        import_error = true;
        msg = PQerrorMessage (conn);
      }

  // if we started the transaction, commit it even in case of import failure
  bool commit_error = false;
  if (make_tblock)
    {
      std::string cmd ("commit;");
      Cell params;
      Cell ptypes (1, 0);
      Cell rtypes;
      command c (*(oct_pq_conn.get_rep ()), cmd, params, ptypes, rtypes, fname);

      if (c.good ())
        c.process_single_result ();

      if (! c.good ())
        commit_error = true;
    }

  if (import_error)
    c_verror ("%s: large object import failed: %s",
              fname.c_str (), msg.c_str ());

  if (commit_error)
    c_verror ("%s: could not commit transaction", fname.c_str ());

  if (import_error || commit_error)
    {
      error ("%s failed", fname.c_str ());
      return retval;
    }

  retval = octave_value (octave_uint32 (oid));

  return retval;
}


DEFUN_DLD (pq_lo_export, args, ,
           "-*- texinfo -*-\n\
@deftypefn {Loadable Function} pq_lo_export (@var{connection}, @var{oid}, @var{path})\n\
Exports the large object of Oid @var{oid} in the database associated with @var{connection} to the file @var{path} on the client side. If @var{path} starts with a @code{|}, it is taken as a shell commant to pipe to.\n\
@end deftypefn")
{
  std::string fname ("pq_lo_export");

  octave_value retval;

  if (args.length () != 3 ||
      args(0).type_id () != octave_pq_connection::static_type_id ())
    {
      print_usage ();

      return retval;
    }

  std::string path;
  CHECK_ERROR (path = args(2).string_value (), retval,
               "%s: third argument can not be converted to a string",
               fname.c_str ());

  bool to_pipe = false;
  if (! path.empty () && path[0] == '|')
    {
      unsigned int pos;
      if ((pos = path.find_first_not_of (" \t\n\r\f", 1))
          == std::string::npos)
        {
          error ("%s: no command found to pipe to", fname.c_str ());

          return retval;
        }
      path.erase (0, pos);

      to_pipe = true;
    }

  Oid oid;
  CHECK_ERROR (oid = args(1).uint_value (), retval,
               "%s: second argument can not be converted to an oid",
               fname.c_str ());

  const octave_base_value& rep = (args(0).get_rep ());

  const octave_pq_connection &oct_pq_conn =
    dynamic_cast<const octave_pq_connection&> (rep);

  PGconn *conn = oct_pq_conn.get_rep ()->octave_pq_get_conn ();

  if (! conn)
    {
      error ("%s: connection not open", fname.c_str ());
      return retval;
    }

  bool make_tblock = false;
  switch (PQtransactionStatus (conn))
    {
    case PQTRANS_IDLE:
      make_tblock = true;
      break;
    case PQTRANS_INTRANS:
      break;
    case PQTRANS_INERROR:
      error ("%s: can't manipulate large objects within a failed transaction block",
             fname.c_str ());
      return retval;
    case PQTRANS_UNKNOWN:
      error ("%s: connection is bad", fname.c_str ());
      return retval;
    default: // includes PQTRANS_ACTIVE
      error ("%s: unexpected connection state", fname.c_str ());
      return retval;
    }

  if (make_tblock)
    {
      std::string cmd ("begin;");
      Cell params;
      Cell ptypes (1, 0);
      Cell rtypes;
      command c (*(oct_pq_conn.get_rep ()), cmd, params, ptypes, rtypes, fname);

      if (c.good ())
        c.process_single_result ();

      if (! c.good ())
        {
          error ("%s: could not begin transaction", fname.c_str ());
          return retval;
        }
    }

  bool export_error = false;
  std::string msg;

  if (to_pipe)
    {
      lo_to_pipe tp (*(oct_pq_conn.get_rep ()), oid, path.c_str (), make_tblock, msg);

      make_tblock = false; // commit handled by destructor of lo_to_pipe

      if (! tp.valid ())
        export_error = true;
    }
  else
    if (lo_export (conn, oid, path.c_str ()) == -1)
      {
        export_error = true;
        msg = PQerrorMessage (conn);
      }

  // if we started the transaction, commit it even in case of export failure
  bool commit_error = false;
  if (make_tblock)
    {
      std::string cmd ("commit;");
      Cell params;
      Cell ptypes (1, 0);
      Cell rtypes;
      command c (*(oct_pq_conn.get_rep ()), cmd, params, ptypes, rtypes, fname);

      if (c.good ())
        c.process_single_result ();

      if (! c.good ())
        commit_error = true;
    }

  if (export_error)
    c_verror ("%s: large object export failed: %s",
              fname.c_str (), msg.c_str ());

  if (commit_error)
    c_verror ("%s: could not commit transaction", fname.c_str ());

  if (export_error || commit_error)
    error ("%s failed", fname.c_str ());

  return retval;
}


DEFUN_DLD (pq_lo_unlink, args, ,
           "-*- texinfo -*-\n\
@deftypefn {Loadable Function} pq_lo_unlink (@var{connection}, @var{oid})\n\
Removes the large object of Oid @var{oid} from the database associated with @var{connection}.\n\
@end deftypefn")
{
  std::string fname ("pq_lo_unlink");

  octave_value retval;

  if (args.length () != 2 ||
      args(0).type_id () != octave_pq_connection::static_type_id ())
    {
      print_usage ();

      return retval;
    }

  Oid oid;
  CHECK_ERROR (oid = args(1).uint_value (), retval,
               "%s: second argument can not be converted to an oid",
               fname.c_str ());

  const octave_base_value& rep = (args(0).get_rep ());

  const octave_pq_connection &oct_pq_conn =
    dynamic_cast<const octave_pq_connection&> (rep);

  PGconn *conn = oct_pq_conn.get_rep ()->octave_pq_get_conn ();

  if (! conn)
    {
      error ("%s: connection not open", fname.c_str ());
      return retval;
    }

  bool make_tblock = false;
  switch (PQtransactionStatus (conn))
    {
    case PQTRANS_IDLE:
      make_tblock = true;
      break;
    case PQTRANS_INTRANS:
      break;
    case PQTRANS_INERROR:
      error ("%s: can't manipulate large objects within a failed transaction block",
             fname.c_str ());
      return retval;
    case PQTRANS_UNKNOWN:
      error ("%s: connection is bad", fname.c_str ());
      return retval;
    default: // includes PQTRANS_ACTIVE
      error ("%s: unexpected connection state", fname.c_str ());
      return retval;
    }

  if (make_tblock)
    {
      std::string cmd ("begin;");
      Cell params;
      Cell ptypes (1, 0);
      Cell rtypes;
      command c (*(oct_pq_conn.get_rep ()), cmd, params, ptypes, rtypes, fname);

      if (c.good ())
        c.process_single_result ();

      if (! c.good ())
        {
          error ("%s: could not begin transaction", fname.c_str ());
          return retval;
        }
    }

  bool unlink_error = false;
  std::string msg;

  if (lo_unlink (conn, oid) == -1)
    {
      unlink_error = true;
      msg = PQerrorMessage (conn);
    }

  // if we started the transaction, commit it even in case of unlink failure
  bool commit_error = false;
  if (make_tblock)
    {
      std::string cmd ("commit;");
      Cell params;
      Cell ptypes (1, 0);
      Cell rtypes;
      command c (*(oct_pq_conn.get_rep ()), cmd, params, ptypes, rtypes, fname);

      if (c.good ())
        c.process_single_result ();

      if (! c.good ())
        commit_error = true;
    }

  if (unlink_error)
    c_verror ("%s: large object unlink failed: %s",
              fname.c_str (), msg.c_str ());

  if (commit_error)
    c_verror ("%s: could not commit transaction", fname.c_str ());

  if (unlink_error || commit_error)
    error ("%s failed", fname.c_str ());      

  return retval;
}
