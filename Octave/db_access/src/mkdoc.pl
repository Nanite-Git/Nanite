#! /usr/bin/perl -w
#
# Copyright (C) 2012-2013 Rik Wehbring
#
# This file is part of Octave.
#
# Octave is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at
# your option) any later version.
#
# Octave is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# along with Octave; see the file COPYING.  If not, see
# <http://www.gnu.org/licenses/>.

# copied from Octave and slightly modified by Olaf Till

unless (@ARGV >= 1) { die "Usage: $0 m_filename1 ..." ; }

print <<__END_OF_MSG__;
### This file is generated automatically by the packages `mkdoc.pl'.

__END_OF_MSG__

MFILE: foreach $full_fname (@ARGV)
{
  next MFILE unless ( $full_fname =~ m{([^/]+)\.m$} );
  $fcn = $1;

  @help_txt = gethelp ($fcn, $full_fname);
  next MFILE if ($help_txt[0] eq "");

  print "\x{1d}$fcn\n";
  print "\@c $fcn $full_fname\n";

  foreach $_ (@help_txt)
  {
    s/^\s+\@/\@/ unless $in_example;
    s/^\s+\@group/\@group/;
    s/^\s+\@end\s+group/\@end group/;
    $in_example = (/\s*\@example\b/ .. /\s*\@end\s+example\b/);
    print $_;
  }
}

################################################################################
# Subroutines
################################################################################
sub gethelp
{
  ($fcn, $fname) = @_[0..1]; 
  open (FH, $fname) or return "";

  do
  {
    @help_txt = ();

    ## Advance to non-blank line
    while (defined ($_ = <FH>) and /^\s*$/) {;}

    if (! /^\s*(?:#|%)/ or eof (FH))
    {
      ## No comment block found.  Return empty string
      close (FH);
      return "";
    }

    ## Extract help text stopping when comment block ends
    do
    {
      ## Remove comment characters at start of line
      s/^\s*(?:#|%){1,2} ?//;
      push (@help_txt, $_);
    } until (! defined ($_ = <FH>) or ! /^\s*(?:#|%)/);

  } until ($help_txt[0] !~ /^(?:Copyright|Author)/); 

  close (FH);

  return @help_txt;
}
