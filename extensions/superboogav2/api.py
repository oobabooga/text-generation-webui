"""
This module is responsible for the VectorDB API. It currently supports:
* DELETE api/v1/clear
    - Clears the whole DB.
* POST api/v1/add
    - Add some corpus to the DB. You can also specify metadata to be added alongside it.
* POST api/v1/delete
    - Delete specific records with given metadata.
* POST api/v1/get
    - Get results from chromaDB.
"""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from threading import Thread

from modules import shared
from modules.logging_colors import logger

from .chromadb import ChromaCollector
from .data_processor import process_and_add_to_collector

import extensions.superboogav2.parameters as parameters


class CustomThreadingHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address, RequestHandlerClass, collector: ChromaCollector, bind_and_activate=True):
        self.collector = collector
        super().__init__(server_address, RequestHandlerClass, bind_and_activate)

    def finish_request(self, request, client_address):
        self.RequestHandlerClass(request, client_address, self, self.collector)


class Handler(BaseHTTPRequestHandler):
    def __init__(self, request, client_address, server, collector: ChromaCollector):
        self.collector = collector
        super().__init__(request, client_address, server)


    def _send_412_error(self, message):
        self.send_response(412)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        response = json.dumps({"error": message})
        self.wfile.write(response.encode('utf-8'))


    def _send_404_error(self):
        self.send_response(404)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        response = json.dumps({"error": "Resource not found"})
        self.wfile.write(response.encode('utf-8'))


    def _send_400_error(self, error_message: str):
        self.send_response(400)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        response = json.dumps({"error": error_message})
        self.wfile.write(response.encode('utf-8'))
        

    def _send_200_response(self, message: str):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()

        if isinstance(message, str):
            response = json.dumps({"message": message})
        else:
            response = json.dumps(message)

        self.wfile.write(response.encode('utf-8'))


    def _handle_get(self, search_strings: list[str], n_results: int, max_token_count: int, sort_param: str):
        if sort_param == parameters.SORT_DISTANCE:
            results = self.collector.get_sorted_by_dist(search_strings, n_results, max_token_count)
        elif sort_param == parameters.SORT_ID:
            results = self.collector.get_sorted_by_id(search_strings, n_results, max_token_count)
        else: # Default is dist
            results = self.collector.get_sorted_by_dist(search_strings, n_results, max_token_count)
        
        return {
            "results": results
        }

        
    def do_GET(self):
        self._send_404_error()


    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            body = json.loads(self.rfile.read(content_length).decode('utf-8'))

            parsed_path = urlparse(self.path)
            path = parsed_path.path
            query_params = parse_qs(parsed_path.query)

            if path in ['/api/v1/add', '/api/add']:
                corpus = body.get('corpus')
                if corpus is None:
                    self._send_412_error("Missing parameter 'corpus'")
                    return
                
                clear_before_adding = body.get('clear_before_adding', False)
                metadata = body.get('metadata')
                process_and_add_to_collector(corpus, self.collector, clear_before_adding, metadata)
                self._send_200_response("Data successfully added")

            elif path in ['/api/v1/delete', '/api/delete']:
                metadata = body.get('metadata')
                if corpus is None:
                    self._send_412_error("Missing parameter 'metadata'")
                    return
                
                self.collector.delete(ids_to_delete=None, where=metadata)
                self._send_200_response("Data successfully deleted")

            elif path in ['/api/v1/get', '/api/get']:
                search_strings = body.get('search_strings')
                if search_strings is None:
                    self._send_412_error("Missing parameter 'search_strings'")
                    return
                
                n_results = body.get('n_results')
                if n_results is None:
                    n_results = parameters.get_chunk_count()
                
                max_token_count = body.get('max_token_count')
                if max_token_count is None:
                    max_token_count = parameters.get_max_token_count()
                
                sort_param = query_params.get('sort', ['distance'])[0]

                results = self._handle_get(search_strings, n_results, max_token_count, sort_param)
                self._send_200_response(results)

            else:
                self._send_404_error()
        except Exception as e:
            self._send_400_error(str(e))


    def do_DELETE(self):
        try:
            parsed_path = urlparse(self.path)
            path = parsed_path.path
            query_params = parse_qs(parsed_path.query)

            if path in ['/api/v1/clear', '/api/clear']:
                self.collector.clear()
                self._send_200_response("Data successfully cleared")
            else:
                self._send_404_error()
        except Exception as e:
            self._send_400_error(str(e))


    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()


    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', '*')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()


class APIManager:
    def __init__(self, collector: ChromaCollector):
        self.server = None
        self.collector = collector
        self.is_running = False

    def start_server(self, port: int):
        if self.server is not None:
            print("Server already running.")
            return

        address = '0.0.0.0' if shared.args.listen else '127.0.0.1'
        self.server = CustomThreadingHTTPServer((address, port), Handler, self.collector)

        logger.info(f'Starting chromaDB API at http://{address}:{port}/api')

        Thread(target=self.server.serve_forever, daemon=True).start()

        self.is_running = True

    def stop_server(self):
        if self.server is not None:
            logger.info(f'Stopping chromaDB API.')
            self.server.shutdown()
            self.server.server_close()
            self.server = None
            self.is_running = False

    def is_server_running(self):
        return self.is_running