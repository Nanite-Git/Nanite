#!/usr/bin/perl -w
#
# Copyright (C) 2012-2014 Rik Wehbring
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

# Validate program call
die "usage: munge-texi DOCSTRING-FILE ... < file" if (@ARGV < 1);

# Constant patterns
$doc_delim = qr/^\x{1d}/;
$tex_delim = qr/\Q-*- texinfo -*-\E/;
$comment_line = qr/^\s*(?:$|#)/;
# Pre-declare hash size for efficiency
keys(%help_text) = 1800;

################################################################################
# Load DOCSTRINGS into memory while expanding @seealso references
foreach $DOCSTRING_file (@ARGV)
{
  open (DOCFH, $DOCSTRING_file) or die "Unable to open $DOCSTRING_file\n";

  # Skip comments
  while (defined ($_ = <DOCFH>) and /$comment_line/o) {;}

  # Validate DOCSTRING file format
  die "invalid doc file format\n" if (! /$doc_delim/o);
  
  do 
  {
    s/\s*$//;   # strip EOL character(s)
    $symbol = substr ($_,1);
    $docstring = extract_docstring ();
    if ($help_text{$symbol})
    {
      warn "ignoring duplicate entry for $symbol\n";
    }
    else
    {
      $help_text{$symbol} = $docstring;
    }

  } while (! eof);

}

################################################################################
# Process .txi to .texi by expanding @DOCSTRING, @EXAMPLEFILE macros

# Add warning header
print '@c This file is generated automatically by the packages munge-texi.pl.',"\n\n";

# parse ../DESCRIPTION to get package version
open (FH, "../DESCRIPTION") or die "unable to open file ../DESCRIPTION\n";
while (<FH>)
{
  # this represents the current logic of pkg
  if (/^Version\s*:\s*([^\s]+)/i)
  {
    $version = $1;
  }
}
close (FH);
defined ($version) or die "unable to find version string in ../DESCRIPTION\n";

TXI_LINE: while (<STDIN>)
{
  if (/^\s*\@DOCSTRING\((\S+)\)/)
  {
    $func = $1;
    $docstring = $help_text{$func};
    if (! $docstring)
    {
      warn "no docstring entry for $func\n";
      next TXI_LINE;
    }

    $func =~ s/^@/@@/;   # Texinfo uses @@ to produce '@'
    $docstring =~ s/^$tex_delim$/\@anchor{XREF$func}/m;
    print $docstring,"\n";

    next TXI_LINE;
  }
  if (/^\s*\@DOCSTRINGVERBATIM\((\S+)\)/)
  {
    $func = $1;
    $docstring = $help_text{$func};
    if (! $docstring)
    {
      warn "no docstring entry for $func\n";
      next TXI_LINE;
    }

    $func =~ s/^@/@@/;   # Texinfo uses @@ to produce '@'
    print '@anchor',"{XREF$func}\n";
    $docstring =~ s/^@[cC] .*?\n//;
    print '@verbatim',"\n";
    print $docstring,"\n";
    print '@end verbatim',"\n";

    next TXI_LINE;
  }
  if (/^\s*\@EXAMPLEFILE\((\S+)\)/)
  {
    $fname = "$1";
    print '@verbatim',"\n";
    open (EXAMPFH, $fname) or die "unable to open example file $fname\n";
    while (<EXAMPFH>) 
    { 
      print $_;
      print "\n" if (eof and substr ($_, -1) ne "\n");
    }
    close (EXAMPFH);
    print '@end verbatim',"\n\n";

    next TXI_LINE;
  }

  # replace @PACKAGEVERSION with version string
  s/\@PACKAGEVERSION/$version/o;

  # pass ordinary lines straight through to output
  print $_;
}


################################################################################
# Subroutines 
################################################################################
sub extract_docstring
{
  my ($docstring, $arg_list, $func_list, $repl, $rest_of_line);
  
  $cut = 0;

  DOC_LINE: while (defined ($_ = <DOCFH>) and ! /$doc_delim/o)
  {
    # drop text marked for cutting out
    if (/^\s*\@[cC]\s*BEGIN_CUT_TEXINFO\s*$/)
    {
      die "BEGIN_CUT_TEXINFO not matched by END_CUT_TEXINFO" if ($cut);
      $cut = 1;
      next DOC_LINE;
    }
    if (/^\s*\@[cC]\s*END_CUT_TEXINFO\s*$/)
    {
      die "END_CUT_TEXINFO without BEGIN_CUT_TEXINFO" if (! $cut);
      $cut = 0;
      next DOC_LINE;
    }
    if ($cut)
    {
      next DOC_LINE;
    }
    # expand any @seealso references
    if (m'^@seealso\s*{')
    {
      # Join multiple lines until full macro body found
      while (! /}/m) { $_ .= <DOCFH>; }

      ($arg_list, $rest_of_line) = m'^@seealso\s*{(.*)}(.*)?'s;
     
      $func_list = $arg_list;
      $func_list =~ s/\s+//gs;
      $repl = "";
      foreach $func (split (/,/, $func_list))
      {
        $func =~ s/^@/@@/;   # Texinfo uses @@ to produce '@'
        $repl .= "\@ref{XREF$func,,$func}, ";
      }
      substr($repl,-2) = "";   # Remove last ', ' 
      $_ = "\@seealso{$repl}$rest_of_line";
    }

    $docstring .= $_;
  }

  return $docstring;
}

