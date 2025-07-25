#!/usr/bin/env python3
# -*- coding: utf8 -*-

#    liberate - Download and/or convert multiple files to Ogg Vorbis
#    Copyright © 2017 Nichlas Severinsen
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see http://www.gnu.org/licenses/.

import os
import sys
import argparse
import colorama
import youtube_dl


_ffmpeg_opts = """
    ffmpeg \
      -i "%s" \
      -v panic \
      -af dynaudnorm \
      -af silenceremove=1:0:-50dB \
      -vn \
      -c:a libvorbis \
      -q:a 5 \
      -f ogg \
      "%s".ogg
  """


def info(msg, filename=''):
  """Print a simple info message.

  Args:
    msg (str): message to print
    filename (str): optional filename to specify
  """
  y = colorama.Fore.WHITE
  r = colorama.Style.RESET_ALL
  print(y + "INFO: %s '%s'" % (msg, filename) + r)
  

def warning(msg, filename=''):
  """Print a simple warning message.

  Args:
    msg (str): message to print
    filename (str): optional filename to specify
  """
  y = colorama.Fore.YELLOW
  r = colorama.Style.RESET_ALL
  print(y + "WARNING: %s '%s'" % (msg, filename) + r)


def file_to_ogg(input_file, remove=False, ffmpeg_opts=_ffmpeg_opts):
  """Convert a single file to ogg using ffmpeg_opts with os.system

  Args:
    input_file (str): file or path to file to convert
    remove (bool): whether or not to remove the original file. False by default
    ffmpeg_opts (str): ffmpeg command and opts
  """
  output_file, file_extension = os.path.splitext(input_file)
  if file_extension == '.ogg':
    warning('conversion ignored, input_file is already .ogg;', input_file)
    return
  os.system(ffmpeg_opts % (input_file.replace('"','\"'), output_file.replace('"','\"')))
  if remove:
    os.remove(input_file)


def try_yt_dl(yt_dl, item, remove_flag):
  """Try to convert item downloaded by yt_dl

  Args:
    item (str): file or path to file to convert
  """
  try:
    if 'http' not in item['url']:
      item = yt_dl.extract_info('https://www.youtube.com/watch?v=' + item['url'], download=False )
    itemname = '%s-%s' % (item['title'].replace('/','_'), item['id'])
    filename = '%s.%s' % (itemname, item['ext'])
    outfile =  '%s.ogg' % (itemname)
    if outfile in os.listdir():
      info('file has already been downloaded and converted', outfile)
      return
    yt_dl.download([item['webpage_url']])
    file_to_ogg(filename, remove_flag)
  except (KeyError, youtube_dl.utils.DownloadError):
    warning('failed to download and/or convert', item['title'])


if __name__ == '__main__':
  colorama.init()
  
  yt_dl_opts = {
    'format': 'bestaudio',
    'ignore-errors': True,
    'quiet': True
  }

  yt_dl = youtube_dl.YoutubeDL(yt_dl_opts)

  parser = argparse.ArgumentParser()
  parser.add_argument('-r', '--remove', dest='remove_flag', help='Remove files after conversion.', action='store_true')
  args = parser.parse_known_args()  

  for argument in args[1]:
    if os.path.isfile(argument): # Single existing file
      file_to_ogg(argument, args[0].remove_flag)
      continue

    if os.path.isdir(argument): # Single existing directory (not recursive!)    
      for input_file in os.listdir(argument):
        file_to_ogg(argument+input_file, args[0].remove_flag)
      continue

    yt_dl.params['extract_flat'] = True
 
    # If not a file and not a directory, try to get info with yt_dl
    try:
      yt_dl_result = yt_dl.extract_info(argument, download=False)
      if yt_dl_result['extractor'] == 'youtube:channel':
        yt_dl_result = yt_dl.extract_info(yt_dl_result['url'], download=False)
    except (KeyError, youtube_dl.utils.DownloadError):
      warning('youtube-dl failed to download', argument)
      continue

    if 'entries' in yt_dl_result: # If yt_dl found a playlist or several items
      for item in yt_dl_result['entries']:
        try_yt_dl(yt_dl, item, args[0].remove_flag)
    else: # yt_dl found one item
      try_yt_dl(yt_dl, yt_dl_result, args[0].remove_flag)
