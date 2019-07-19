import requests
import pandas as pd
import json
import os
import numpy as np
from difflib import SequenceMatcher
import time
from apis.core_api import BaseAPI, multitry
from utils.common_utils import *
from datetime import datetime


class SOSAPI(BaseAPI): 

    HOST= 'http://test.sos.api.ds.credibly.com'
    PATH='/api/sos'
        
    def initialize(self,user_query):

        conf = user_query['conf']['sos']
        self.HOST = conf['HOST']
        self.PATH = conf['PATH']
        self.query=user_query['business']
        self.req_input={"name":user_query['business']['name'], "state":user_query['business']['state']}
        self.score=[]
        self.url = '{0}{1}'.format(self.HOST, self.PATH)
        self.headers={'Content-Type': 'application/json','Accept': 'application/json'}
        self.attempts_allowed = conf['max_retry']
        self.timeout = conf['timeout'] 
        self.auth_result['auth_tokens'] = conf
        
    def form_requests(self,user_query):
        headers =  {'Content-Type': 'application/json','Accept': 'application/json'}
        requests = []
        if user_query['business']['state'] in user_query.get('conf',{}).get('sos_states',[]):
            requests.append({"name":user_query['business']['name'], "state":user_query['business']['state']})
            requests.append({"name":user_query['business']['dba'], "state":user_query['business']['state']})
        return headers,requests


    def get_eligible_responses(self, responses):
        eligible_responses = []       
        for resp in responses:
            if resp is not None and resp.get('scores',None) is not None and set(resp['scores'].keys()) >= set(['NAME_EDIT_DIST','ADDR_EDIT_DIST','STNUM_MATCH','PHONE_MATCH','ZIP_MATCH','CITY_MATCH']):
                max_name_dist=.3
                max_addr_dist=.4
                best_match_score = resp['scores']
                city_match2 = best_match_score['CITY_MATCH']

                addr_match = (city_match2 | (best_match_score['ADDR_EDIT_DIST'] <= max_addr_dist))
                addr_match2 = ((best_match_score['ADDR_EDIT_DIST'] <= .2) | best_match_score['STNUM_MATCH'])

                phone_match = best_match_score['PHONE_MATCH']
                zip_match = best_match_score['ZIP_MATCH']
                similar_name = best_match_score['NAME_EDIT_DIST'] <= max_name_dist
                name_fuzzy = best_match_score['NAME_EDIT_DIST'] <= .5

                # ZIP matches and name is similar
                best_match_criteria = ((zip_match | addr_match) & (similar_name | phone_match)) | (addr_match2 & (name_fuzzy | phone_match))
                
                if best_match_criteria:
                    eligible_responses.append(resp)
        return eligible_responses
     
    
    
    def parse_response(self, user_query, api_response):

        responses = []
        resp=api_response
        if  len(resp) > 0:
            for item in resp:
                match = item.copy()
                scores = {}

                scores['PHONE_MATCH'] = False

                scores['NAME_EDIT_DIST'] = name_edit_dist([match.get('companyName',None)],[user_query['business'].get('dba',None),user_query['business'].get('name',None)])

                scores['ADDR_EDIT_DIST'] = addr_edit_dist(match.get('address',{}).get('address1',None),user_query['business'].get('address', None))
                scores['ZIP_MATCH'] = format_zip(match.get('address',{}).get('zip',None)) == format_zip(user_query['business'].get('zip', None))
                scores['STNUM_MATCH'] = strt_num_match(match.get('address', {}).get('address1',None),user_query['business'].get('address', None))
                scores['CITY_MATCH'] = city_match(match.get('address',{}).get('city',None),user_query['business'].get('city', None))
                NAME_EDIT_DIST = scores['NAME_EDIT_DIST']
                PHONE_MATCH = scores['PHONE_MATCH']
                ADDR_EDIT_DIST = scores['ADDR_EDIT_DIST']
                scores['MATCH_SCORE'] = 10 * ((((NAME_EDIT_DIST if NAME_EDIT_DIST < 0.1 else 0.1) if PHONE_MATCH else NAME_EDIT_DIST) * (ADDR_EDIT_DIST if ADDR_EDIT_DIST >= 0 else 0.3)) ** 0.5)

                if scores['MATCH_SCORE'] < 10: # 10 is a cut-off score
                    responses.append({'response':match,'scores':scores})


        return self.get_eligible_responses(responses)

    
    def score_and_sort_matches(self, responses):
        sorted_responses = sorted(responses, key=lambda k: k['scores']['MATCH_SCORE'])
        if len(sorted_responses) > 1 and sorted_responses[0]['scores']['MATCH_SCORE'] == sorted_responses[1]['scores']['MATCH_SCORE']:
            min_score = sorted_responses[0]['scores']['MATCH_SCORE']
            min_score_responses = []
            conjugated_scores = []
            for item in sorted_responses:
                if item['scores']['MATCH_SCORE'] == min_score:
                    min_score_responses.append(item)
                    conjugated_score = item['scores']['MATCH_SCORE'] 
                    if item['scores']['ADDR_EDIT_DIST'] >= 0:
                        conjugated_score += item['scores']['ADDR_EDIT_DIST']
                    else:
                        conjugated_score += 5 # when dist is np.nan ie infinite
                    if item['scores']['NAME_EDIT_DIST'] >= 0:
                        conjugated_score += item['scores']['NAME_EDIT_DIST']
                    else:
                        conjugated_score += 5 # when dist is np.nan ie infinite
                    if item['scores']['ZIP_MATCH']:
                        conjugated_score += 0
                    else:
                        conjugated_score += 1

                        
                    if item['scores']['CITY_MATCH']:
                        conjugated_score += 0
                    else:
                        conjugated_score += 1
                    conjugated_scores.append(conjugated_score)
            return min_score_responses[conjugated_scores.index(min(conjugated_scores))]
            
        else:
            return sorted_responses[0]


    @multitry
    def call_api(self, user_query):
        try:
            api_calls = []
            self.error_string = ''
            self.initialize(user_query)
            headers, api_requests=self.form_requests(user_query)
            url = '{0}{1}?'.format(self.HOST, self.PATH)
            headers = headers or {}
            best_match = None
            if self.verbose:
                print('Querying {0} ...'.format(url))
            status = 'OK'
            for api_request in api_requests:
                api_call = {}
                api_request = api_request or {}
                consumer_id = None
                if 'consumer_id' in api_request:
                    consumer_id = api_request.pop('consumer_id')
                #result = requests.get(url, headers=headers, params=api_request, timeout=self.timeout)
                result = self.post_request(headers, api_request)
                try:
                    responses = self.parse_response(user_query, result.json())
                except Exception as e:
                    #print(e)
                    #print(traceback.format_exc())
                    responses = {'content':result.content}
                    status = 'Error'
                api_call['request'] = self.clean_request(api_request)
                api_call['response'] = responses
                api_call['status'] = status
                api_call['timestamp'] = datetime.now().isoformat()
                if consumer_id is not None:
                    api_call['consumer_id'] = consumer_id
                api_calls.append(api_call)
                best_match = self.get_best_match(responses)
                if best_match != None:
                    break
            return {'best_match':best_match,'raw_data':api_calls,'status':status}
            #return best_match, api_calls
        except Exception as e:
            self.error_string += str(e)
            log_error(user_query['header']['request_id'], 'Error in calling' + str(self) + str(e), traceback)
            return {'best_match':None,'raw_data':api_calls,'status':'Error','message':self.error_string, 'timestamp':datetime.now().isoformat()}