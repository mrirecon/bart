# Copyright 2023. Christian Tönnes.
# All rights reserved. Use of this source code is governed by
# a BSD-style license which can be found in the LICENSE file.
# 
# Authors:
# 2023 Christian Tönnes <christian.toennes@keks.li>
#

# pip install libsass watchdog
from http.server import SimpleHTTPRequestHandler, HTTPServer
import mimetypes

class RequestHandler(SimpleHTTPRequestHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="./wwwroot", **kwargs)

    if not mimetypes.inited:
        mimetypes.init() # try to read system mime.types
    extensions_map = mimetypes.types_map.copy()
    extensions_map.update({
        '': 'application/octet-stream', # Default
        '.py': 'text/plain',
        '.c': 'text/plain',
        '.h': 'text/plain',
        '.js': 'application/javascript',
        })    

handler_class = RequestHandler
httpd = HTTPServer(('', 8000), handler_class)

httpd.serve_forever()