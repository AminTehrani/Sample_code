"""
Handles the event sequence for preapproval process. Interacts with api_hub, scoring, offers and rule engine.
"""
# TODO implement logging
from utils.db_helper import *
from utils.pq_dupe import *
import random
from email_validator import validate_email, EmailNotValidError, EmailSyntaxError
from apis.api_hub import call_apis, get_api_summary
from preapproval.RuleEngine import run_rules
from rcs_pylib.run_score import run_end2end
from lm_pylib.pm_knn import load_knn, run_knn
# from pm_knn import run_knn
from datetime import datetime
from utils.common_utils import *
import json
import math
import pandas as pd
import numpy as np
from preapproval.exceptions import ExperianFailure
import time


pd.options.mode.chained_assignment = None
rule_dynamic_mapping = pd.read_csv('conf/rule_dynamic_text.csv',dtype=object)
industry_naics_mapping = pd.read_csv('conf/Industry_NAICS_SIC_mapping.csv',dtype=object)
industry_naics_mapping = industry_naics_mapping.apply(lambda s: s.strip() if isinstance(s,str) else s)
industry_naics_mapping['Industry'] = industry_naics_mapping['Industry'].apply(lambda s: s.upper() if isinstance(s,str) else s)
valid_industry = list(industry_naics_mapping['Industry'])

def scale_raw_score(raw_score,conf):
    """This is to make score correelated with recap
    Args:
        raw_score: raw_score from scoring service
    Returns:
        score: scaled to align with recap
    """
    if isinstance(raw_score, str):
        raw_score = float(raw_score)
    return  eval(conf.get('adjusted_score',{}).get('value','raw_score'))

#@timed
def generate_rule_input(user_query, api_calls, api_summary,scoring_results,db_info, conf,update, valid_email):
    """Create input to run the rules. Consumer specific rules are sent as list with an item for each party.
    Args:
        user_query: user input
        api_calls: output of api calls
        api_summary: summary of api_calls
        scoring_results: output from scoring
        db_info: results from validation against various datasource in datascience databse 
    Returns:
        dict: rules input
    """
    rule_input = {}
    
    if db_info != None and len(db_info) > 0:
        #if 'mcc_hit' in db_info:
        #    rule_input['mcc_hit'] = db_info['mcc_hit']
        fields_to_ignore = conf.get('bad_merchant_search_ignore_fields',[])
        if not isinstance(fields_to_ignore,list):
            fields_to_ignore = []
        if 'sbfa_hit' in db_info:
            sbfa_resp = db_info['sbfa_hit']
            if isinstance(sbfa_resp,dict) and 'hit' in sbfa_resp and 'matches' in sbfa_resp:
                rule_input['sbfa_hit'] = False
                for match in sbfa_resp['matches']:
                    if isinstance(match,dict):
                        matching_fields = match.get('matching_fields',[])
                        if isinstance(matching_fields, list) and len(matching_fields) == 1 and matching_fields[0] in fields_to_ignore:
                            continue
                        rule_input['sbfa_hit'] = True
        if 'bad_actor' in db_info: # - this repalces mcc, rcc col and rc wo
            bad_actor_resp = db_info['bad_actor']
            rule_input['credibly_bmhit'] = False
            rule_input['bizfi_bmhit'] = False
            if isinstance(bad_actor_resp,dict) and 'hit' in bad_actor_resp and 'matches' in bad_actor_resp and isinstance(bad_actor_resp['matches'],list):
                for match in bad_actor_resp['matches']:
                    if isinstance(match,dict) and 'sources' in match and isinstance(match['sources'], list) and len(set(match['sources']) & set(['credibly_co', 'credibly_collections', 'credibly_dnr30', 'credibly_fraud'])) > 0:
                        matching_fields = match.get('matching_fields',[])
                        if isinstance(matching_fields, list) and len(matching_fields) == 1 and matching_fields[0] in fields_to_ignore:
                            continue
                        
                        rule_input['credibly_bmhit'] = True
                    elif isinstance(match,dict) and 'sources' in match and isinstance(match['sources'], list) and len(set(match['sources']) & set(['bizfi_bad_merchant', 'bizfi_co', 'bizfi_collections'])) > 0:
                        matching_fields = match.get('matching_fields',[])
                        if isinstance(matching_fields, list) and len(matching_fields) == 1 and matching_fields[0] in fields_to_ignore:
                            continue
                        rule_input['bizfi_bmhit'] = True
                    elif isinstance(match,dict) and 'sources' in match and isinstance(match['sources'], list) and len(set(match['sources']) & set(['credibly_fraud'])) > 0:
                        matching_fields = match.get('matching_fields',[])
                        if isinstance(matching_fields, list) and len(matching_fields) == 1 and matching_fields[0] in fields_to_ignore:
                            continue
                        
                        rule_input['credibly_fraud'] = True
                    elif isinstance(match,dict) and 'sources' in match and\
                    isinstance(match['sources'], list) and\
                    len(set(match['sources']) &\
                             set(['jcap_high_delinquency','jcap_chargeoff'])) > 0:
                        matching_fields = match.get('matching_fields',[])
                        if isinstance(matching_fields, list) and len(matching_fields) == 1 and matching_fields[0] in fields_to_ignore:
                            continue
                        
                        rule_input['jcap_bad'] = True
                    elif isinstance(match,dict) and 'sources' in match and\
                    isinstance(match['sources'], list) and\
                    len(set(match['sources']) & \
                        set(['credibly_terminated_iso',
                             'credibly_suspended_iso',
                             'credibly_rejected_iso'])) > 0:
                        matching_fields = match.get('matching_fields',[])
                        if isinstance(matching_fields, list) and len(matching_fields) == 1 and matching_fields[0] in fields_to_ignore:
                            continue
                        
                        rule_input['credibly_bad_iso'] = True
                    elif isinstance(match,dict) and 'sources' in match and isinstance(match['sources'], list) and len(set(match['sources']) & set(['news'])) > 0:
                        matching_fields = match.get('matching_fields',[])
                        if isinstance(matching_fields, list) and len(matching_fields) == 1 and matching_fields[0] in fields_to_ignore:
                            continue
                        
                        rule_input['news_bmhit'] = True
                        
                
        #if 'rcwo_hit' in db_info:
        #    rule_input['rcwo_hit'] = db_info['rcwo_hit']
        #if 'rccol_hit' in db_info:
        #    rule_input['rccol_hit'] = db_info['rccol_hit']
        if 'high_risk_industries' in db_info:
            rule_input['industry_hr'] = db_info['high_risk_industries']['industry_high_risk']
            rule_input['industry_prohibited'] = db_info['high_risk_industries']['industry_prohibited']
            
        if 'msa_risk' in db_info:
            rule_input['msahr'] = db_info['msa_risk']['is_high_risk']
            rule_input['msa_tier'] = db_info['msa_risk']['tier']
            
        if not update and 'queries_cached_apis' in db_info:
            dupe_bus = False
            dupe_csm = []
            max_dupe_query_days = float(conf.get('max_dupe_query_days',{}).get('value',100000))
            max_dupe_query_ts = time.mktime(date_in_days(-1 * max_dupe_query_days).timetuple())
            if 'business_queries' in db_info['queries_cached_apis'] and isinstance(db_info['queries_cached_apis']['business_queries'],dict):
                
                for item in db_info['queries_cached_apis']['business_queries']:
                    if isNumber(db_info['queries_cached_apis']['business_queries'][item]) and float(db_info['queries_cached_apis']['business_queries'][item]) > max_dupe_query_days:
                        dupe_bus = True
                        break
            if 'consumers_queries' in db_info['queries_cached_apis'] and isinstance(db_info['queries_cached_apis']['consumers_queries'],list):   
                for csm_item in db_info['queries_cached_apis']['consumers_queries']:
                    _dupe_csm = False
                    if isinstance(csm_item,dict):
                        for item in csm_item:
                            if isNumber(csm_item[item]) and float(csm_item[item]) > max_dupe_query_days:
                                _dupe_csm = True
                                break
                    dupe_csm.append(_dupe_csm)
            if len(dupe_csm) == 0:
                for csm_it in user_query.get('consumers',[]):
                    dupe_csm.append(False)
            rule_input['dupe_bus'] = dupe_bus
            rule_input['dupe_csm'] = dupe_csm
                            
    if scoring_results != None and 'result' in scoring_results:
        if 'score_raw' in scoring_results['result']:
            #rule_input['score_raw'] = scale_raw_score(scoring_results['result']['score_raw'])
            rule_input['score'] = math.ceil(scale_raw_score(scoring_results['result'].get('score_raw',.999),conf)* 1000)
            rule_input['score_adjr'] = rule_input['score'] # for now scroe and 
        #if 'score_adjr' in scoring_results['result']:
        #    rule_input['score_adjr'] = scoring_results['result']['score_adjr']
        if 'reason_codes' in scoring_results['result']:
            rule_input['score_factors'] = scoring_results['result']['reason_codes']
            if len(scoring_results['result']['reason_codes']) > 0:
                v = rule_dynamic_mapping[(rule_dynamic_mapping['attribute'] == 'top_score_factor_text') & (rule_dynamic_mapping['reason_code'] == str(scoring_results['result']['reason_codes'][0]))]['value']
                if len(v) > 0:
                    rule_input['top_score_factor_text'] = list(v)[0]
        
    consumers = user_query['consumers']
    business = user_query['business']
    age = []
    ssn_valid = []
    ssn_area = []
    ssn_group = []
    ssn_serial = []
    ssn_invalid_grp = []
    for consumer in consumers:
        if is_valid_format(consumer.get('dob',None),'date'):
            consumer_dob = get_dob_components(consumer['dob'])
            age.append(int(consumer_dob['age_years']))
        ssn_valid.append(is_valid_ssn(consumer.get('ssn',None)))
        ssn_components = get_ssn_components(consumer.get('ssn',''))
        ssn_area.append(ssn_components['area'])
        ssn_group.append(ssn_components['group'])
        ssn_serial.append(ssn_components['serial'])
        ssn_invalid_grp.append(ssn_components['invalid_grp_ranking'])
    
    if len(age) > 0:
        rule_input['age'] = age
    rule_input['ssn_valid'] = ssn_valid
    rule_input['ssn_area'] = ssn_area
    rule_input['ssn_group'] = ssn_group
    rule_input['ssn_serial'] = ssn_serial
    rule_input['ssn_invalid_grp'] = ssn_invalid_grp
    
    fico = []
    csm_hit = []
    cbo = []
    csm_victim_tenure = []
    csm_tenure = []
    csm_inqcnt = []
    csm_pastdue = []
    csm_delqderog = []
    csm_bk = []
    csm_judgment = []
    csm_taxlien = []
    csm_ofac = []
    csm_fs = []
    dmf_hit = []
    csm_col_bal = []
    csm_co_bal = []
    csm_tradecnt = []
    conflicting_bk = []
    for consumer_summary in api_summary.get('exp_csm',[]):
        dmf = False
        if 'fico' in consumer_summary:
            fico.append(consumer_summary['fico'])
            if consumer_summary['fico'] == 9001:
                dmf = True
            if consumer_summary['fico'] >= 300 and consumer_summary['fico'] <= 850:
                csm_hit.append(True)
            else:
                csm_hit.append(False)
        else:
            csm_hit.append(False)
            
        if 'cbo' in consumer_summary:
            cbo.append(consumer_summary['cbo'])
        if 'RecentVictimStatementDate' in consumer_summary:
            csm_victim_tenure.append(days_since(consumer_summary['RecentVictimStatementDate'])/365)
        else:
            csm_victim_tenure.append(np.nan)
        if 'OldestTradeOpenDate' in consumer_summary:
            csm_tenure.append(days_since(consumer_summary['OldestTradeOpenDate'])/365)
        
        if 'InquiriesDuringLast6Months' in consumer_summary:
            csm_inqcnt.append(consumer_summary['InquiriesDuringLast6Months'])
        else:
            csm_inqcnt.append(np.nan)
        if 'PastDueAmount' in consumer_summary:
            csm_pastdue.append(consumer_summary['PastDueAmount'])
        else:
            csm_pastdue.append(np.nan)
        if 'NowDelinquentDerog' in consumer_summary:
            csm_delqderog.append(consumer_summary['NowDelinquentDerog'])
        else:
            csm_delqderog.append(np.nan)
        if 'OpenBankruptcies' in consumer_summary:
            bk = len(consumer_summary['OpenBankruptcies']) > 0
            pacer_bk = False
            pacer_called = False
            if 'pacerBk' in api_summary and len(api_summary['pacerBk']) > 0:
                pacer_called = True
            if 'pacerEpnBk' in api_summary and len(api_summary['pacerEpnBk']) > 0:
                for item in api_summary['pacerEpnBk'][0]:
                    if isinstance(item,dict) and 'open_per_pacer' in item and item['open_per_pacer']:
                        pacer_bk = True
            if (not pacer_called or pacer_bk) and bk:# check if bk is true in experian and pacer
                csm_bk.append(True)
            else:
                csm_bk.append(False)
            if (pacer_called and not pacer_bk and bk):
                conflicting_bk.append(True)
                
        else:
            csm_bk.append(False)
        
        if 'TotalCollectionAmount' in consumer_summary:
            csm_col_bal.append(consumer_summary['TotalCollectionAmount'])
        else:
            csm_col_bal.append(0)
        
        if 'TotalChargeOffAmount' in consumer_summary:
            csm_co_bal.append(consumer_summary['TotalChargeOffAmount'])
        else:
            csm_co_bal.append(0)
        if 'NumberOfAccounts' in consumer_summary:
            csm_tradecnt.append(consumer_summary['NumberOfAccounts'])
        else:
            csm_tradecnt.append(0)
        if 'JudgmentAmount' in consumer_summary:
            csm_judgment.append(consumer_summary['JudgmentAmount'])
        else:
            csm_judgment.append(0)
        if 'TaxLienAmount' in consumer_summary:
            csm_taxlien.append(consumer_summary['TaxLienAmount'])
        else:
            csm_taxlien.append(0)
        if 'MatchesOFAC' in consumer_summary:
            csm_ofac.append(consumer_summary['MatchesOFAC'])
        else:
            csm_ofac.append(False)
        
        if 'FraudShieldIndicators' in consumer_summary:
            if 5 in consumer_summary['FraudShieldIndicators']:
                dmf = True
            csm_fs.append(consumer_summary['FraudShieldIndicators'])
        else:
            csm_fs.append([])
        
        dmf_hit.append(dmf)  
            
    if len(fico) > 0:
        rule_input['fico'] = fico
    if len(csm_hit) > 0:
        rule_input['csm_hit'] = csm_hit
    if len(cbo) > 0:
        rule_input['cbo'] = cbo
    if len(csm_victim_tenure) > 0:
        rule_input['csm_victim_tenure'] = csm_victim_tenure
    if len(csm_tenure) > 0:
        rule_input['csm_tenure'] = csm_tenure
    if len(csm_inqcnt) > 0:
        rule_input['csm_inqcnt'] = csm_inqcnt
    if len(csm_pastdue) > 0:
        rule_input['csm_pastdue'] = csm_pastdue
    if len(csm_delqderog) > 0:
        rule_input['csm_delqderog'] = csm_delqderog
    if len(csm_bk) > 0:
        rule_input['csm_bk'] = csm_bk
    if len(csm_judgment) > 0:
        rule_input['csm_judgment'] = csm_judgment
    if len(csm_taxlien) > 0:
        rule_input['csm_taxlien'] = csm_taxlien
    if len(csm_ofac) > 0:
        rule_input['csm_ofac'] = csm_ofac
    if len(csm_fs) > 0:
        rule_input['csm_fs'] = csm_fs
    if len(dmf_hit) > 0:
        rule_input['dmf_hit'] = dmf_hit
    if len(csm_col_bal) > 0:
        rule_input['csm_colbal'] = csm_col_bal
    if len(csm_co_bal) > 0:
        rule_input['csm_cobal'] = csm_co_bal
    if len(csm_tradecnt) > 0:
        rule_input['csm_tradecnt'] = csm_tradecnt
    if len(conflicting_bk) > 0:
        rule_input['conflicting_bk'] = conflicting_bk
        
    fpd = []
    for pid_summary in api_summary.get('experianPid',[]):
        if 'FPDScore' in pid_summary:
            fpd.append(pid_summary['FPDScore'])
    
    if len(fpd) > 0:
        rule_input['fpd'] = fpd
        
    l2c = []
    for l2c_summary in api_summary.get('l2c',[]):
        if 'score' in l2c_summary and isNumber(l2c_summary['score']):
            l2c.append(l2c_summary['score'])
    
    if len(l2c) > 0:
        rule_input['l2c_score'] = l2c
        
    if 'sos' in api_summary and isinstance(api_summary['sos'],list) and len(api_summary['sos']) > 0:
        if 'sos_active' in api_summary['sos'][0]:
            rule_input['sos_active'] = api_summary['sos'][0]['sos_active']
        else:
            rule_input['sos_active'] = False
        if 'sos_type' in api_summary['sos'][0]:
            rule_input['sos_type'] = api_summary['sos'][0]['sos_type']
        if 'sos_owner' in api_summary['sos'][0]:
            rule_input['sos_owner'] = api_summary['sos'][0]['sos_owner']
    elif user_query['business']['state'] in user_query.get('conf',{}).get('sos_states',[]) and 'sos' not in api_summary.get('failed_apis',{}):
            rule_input['sos_active'] = False
    
    nfd_hit = [] 
    for nfd_consumer_summary in api_summary.get('nfd',[]):
        if 'nfd_value' in nfd_consumer_summary:
            nfd_hit.append(nfd_consumer_summary['nfd_value']) # check if nfd has values 1,2,3
    
    if len(nfd_hit) > 0:
        rule_input['nfd_index'] = nfd_hit
        
    lncrim_hit = [] 
    lncrim_shr = []
    for lncrim_consumer_summary in api_summary.get('ln_crim',[]):
        if 'alerts' in lncrim_consumer_summary:
            lncrim_hit.append(True) 
            alert_text = str(lncrim_consumer_summary['alerts']).lower()
            if 'theft' in alert_text or 'fraud' in alert_text:
                lncrim_shr.append(True)
            else:
                lncrim_shr.append(False)
        else:
            lncrim_hit.append(False)
            lncrim_shr.append(False)
    
    if len(lncrim_hit) > 0:
        rule_input['criminal_hr'] = lncrim_hit
    
    if len(lncrim_shr) > 0:
        rule_input['criminal_shr'] = lncrim_shr
    
    ft_prfreq = [] 
    ft_pdtrd = []
    for ft_consumer_summary in api_summary.get('factor_trust',[]):
        llpdnewopen18M = ft_consumer_summary.get('llpdnewopen18M',0)
        ft_pdtrd.append(llpdnewopen18M)
            
        sldiffpayrollfreqever = ft_consumer_summary.get('sldiffpayrollfreqever',0)
        ft_prfreq.append(sldiffpayrollfreqever)
    if len(ft_prfreq) > 0:
        rule_input['ft_prfreq'] = ft_prfreq
    if len(ft_pdtrd) > 0:
        rule_input['ft_pdtrd'] = ft_pdtrd
        
    
    if len(api_summary.get('exp_ipv2',[])) > 0:
        exp_ipv2 = api_summary['exp_ipv2'][0]
        if 'IntelliscorePlusV2' in exp_ipv2:
            rule_input['ipv2'] = exp_ipv2['IntelliscorePlusV2']
            if isInt(exp_ipv2['IntelliscorePlusV2']) and int(exp_ipv2['IntelliscorePlusV2']) < 90000:
                rule_input['cmc_hit'] = True
            else:
                rule_input['cmc_hit'] = False
        else:
            rule_input['cmc_hit'] = False
        
        if 'fsr' in exp_ipv2:
            rule_input['fsr'] = exp_ipv2['fsr']
        
        if 'FileEstablishDate' in exp_ipv2:
            rule_input['cmc_tenure'] = days_since(exp_ipv2['FileEstablishDate'])/365
        
        
        if 'PubliclyHeldCompany' in exp_ipv2:
            rule_input['is_public'] = exp_ipv2['PubliclyHeldCompany']
        
        if 'TaxLienLiabilityAmount' in exp_ipv2:
            rule_input['cmc_taxlien'] = exp_ipv2['TaxLienLiabilityAmount']
        else:
            rule_input['cmc_taxlien'] = 0
            
        if 'JudgementLiabilityAmount' in exp_ipv2:
            rule_input['cmc_judgment'] = exp_ipv2['JudgementLiabilityAmount']
        else:
            rule_input['cmc_judgment'] = 0
        
        if 'OpenBankruptcies' in exp_ipv2 and len(exp_ipv2['OpenBankruptcies']) > 0:
            rule_input['cmc_bk'] = True
        else:
            rule_input['cmc_bk'] = False
            
        if 'InquiriesCountLast6Months' in exp_ipv2:
            rule_input['cmc_inqcnt'] = exp_ipv2['InquiriesCountLast6Months']
        
        
        if 'OpenCollectionsCount' in exp_ipv2:
            rule_input['cmc_colcnt'] = exp_ipv2['OpenCollectionsCount']
        else:
             rule_input['cmc_colcnt'] = 0
        
        if 'UCCFilings' in exp_ipv2:
            rule_input['ucc_cnt'] = exp_ipv2['UCCFilings']
        else:
            rule_input['ucc_cnt'] = 0
        
        if 'CollectionsBalance' in exp_ipv2:
            rule_input['cmc_colbal'] = exp_ipv2['CollectionsBalance']
        else:
            rule_input['cmc_colbal'] = 0
        
        if 'ChargeOffsBalance' in exp_ipv2:
            rule_input['cmc_cobal'] = exp_ipv2['ChargeOffsBalance']
        else:
            rule_input['cmc_cobal'] = 0
    
    if len(api_summary.get('exp_sbcs',[])) > 0:
        exp_sbcs = api_summary['exp_sbcs'][0]
        
        if 'YearsInBusiness' in exp_sbcs:
            rule_input['exp_yib'] = exp_sbcs['YearsInBusiness']
        
        
        if 'SalesRevenue' in exp_sbcs and isNumber(exp_sbcs['SalesRevenue']) and int(exp_sbcs['SalesRevenue']) > 0:
            rule_input['exp_revenue'] = exp_sbcs['SalesRevenue']
        
        if 'MostRecentUCCDate' in exp_sbcs:
            rule_input['ucc_tenure'] = days_since(exp_sbcs['MostRecentUCCDate'])/365
        else:
            rule_input['ucc_tenure'] = np.nan
            
        if 'BusinessVictimStatement' in exp_sbcs:
            rule_input['cmc_victim'] = exp_sbcs['BusinessVictimStatement']
        else:
            rule_input['cmc_victim'] = False
        
        if 'naics' in exp_sbcs:
            rule_input['exp_naics'] = exp_sbcs['naics']
            
        if 'sic' in exp_sbcs:
            rule_input['exp_sic'] = exp_sbcs['sic']
    
    
    rule_input['tib'] = get_tib_months(business['business_start_date'])/12
    api_domains = []
    api_phones = []
    if isNumber(business.get('deposit_amt',None)):
        rule_input['bd_amt'] = int(business['deposit_amt'])
    
    if isNumber(business.get('revenue',None)):
        rule_input['revenue'] = int(business['revenue'])
    elif 'bd_amt' in rule_input:
        rule_input['revenue'] = rule_input['bd_amt'] * 12
    else:
        rule_input['revenue'] = np.nan
    
    if len(api_summary.get('google',{})) > 0:
        if 'permanently_closed' in api_summary['google'][0]:
            rule_input['google_closed'] = api_summary['google'][0]['permanently_closed']
        else:
            rule_input['google_closed'] = False
        
        if 'website' in api_summary['google'][0]:
            domain = get_domain(api_summary['google'][0]['website'])
            if domain is not None:
                api_domains.append(domain)
        if 'phones' in api_summary['google'][0] and isinstance(api_summary['google'][0]['phones'],list):
            api_phones.extend(api_summary['google'][0]['phones']) 
            
    elif 'google' in api_calls and 'google' not in api_summary.get('failed_apis',{}):
        rule_input['google_closed'] = False
        
    
    if len(api_summary.get('yelp',{})) > 0:
        if 'permanently_closed' in api_summary['yelp'][0]:
            rule_input['yelp_closed'] = api_summary['yelp'][0]['permanently_closed']
        else:
            rule_input['yelp_closed'] = False
        
        if 'phones' in api_summary['yelp'][0] and isinstance(api_summary['yelp'][0]['phones'],list):
            api_phones.extend(api_summary['yelp'][0]['phones'])
            
    elif 'yelp' in api_calls and 'yelp' not in api_summary.get('failed_apis',{}):
        rule_input['yelp_closed'] = False
    
    if len(api_summary.get('bbb',{})) > 0:
        if 'Rating' in api_summary['bbb'][0]:
            rule_input['bbb_rating'] = api_summary['bbb'][0]['Rating']
        if 'business_started_date' in api_summary['bbb'][0]:
            rule_input['bbb_tib'] = get_dob_components(api_summary['bbb'][0]['business_started_date']).get('age_years',0)
        
        if 'website' in api_summary['bbb'][0]:
            domain = get_domain(api_summary['bbb'][0]['website'])
            if domain is not None:
                api_domains.append(domain)
        if 'phones' in api_summary['bbb'][0] and isinstance(api_summary['bbb'][0]['phones'],list):
            api_phones.extend(api_summary['bbb'][0]['phones'])
            
        if 'Inactive' in api_summary['bbb'][0]:
            rule_input['bbb_inactive'] = api_summary['bbb'][0]['Inactive']
    
    if 'bbb' in api_calls and 'bbb' not in api_summary.get('failed_apis',{}):
        if 'bbb_inactive' not in rule_input:
            rule_input['bbb_inactive'] = False
        if 'bbb_rating' not in rule_input:
            rule_input['bbb_rating'] = 'N/A'
        
    
    if len(api_summary.get('d&bDirect',{})) > 0:
        if 'yearlyRevenue' in api_summary['d&bDirect'][0] and isNumber(api_summary['d&bDirect'][0]['yearlyRevenue']):
            rule_input['duns_revenue'] = float(api_summary['d&bDirect'][0]['yearlyRevenue'])
        
        if 'mail_undeliverable' in api_summary['d&bDirect'][0]:
            rule_input['mail_undeliverable'] = api_summary['d&bDirect'][0]['mail_undeliverable']
        
        if 'out_of_business' in api_summary['d&bDirect'][0]:
            rule_input['duns_oob'] = api_summary['d&bDirect'][0]['mail_undeliverable']
        
        if 'phone_disconnected' in api_summary['d&bDirect'][0]:
            rule_input['phone_disconnected'] = api_summary['d&bDirect'][0]['phone_disconnected']
            #rule_input['duns_phone_number'] = api_summary['d&bDirect'][0]['disconnected_phone']
        
        if 'phones' in api_summary['d&bDirect'][0] and isinstance(api_summary['d&bDirect'][0]['phones'],list):
            api_phones.extend(api_summary['d&bDirect'][0]['phones'])
        
        if 'website' in api_summary['d&bDirect'][0]:
            domain = get_domain(api_summary['d&bDirect'][0]['website'])
            if domain is not None:
                api_domains.append(domain)
        
    if len(api_summary.get('facebook',{})) > 0:
        if 'permanently_closed' in api_summary['facebook'][0]:
            rule_input['facebook_closed'] = api_summary['facebook'][0]['permanently_closed']
        
        if 'phones' in api_summary['facebook'][0] and isinstance(api_summary['facebook'][0]['phones'],list):
            api_phones.extend(api_summary['facebook'][0]['phones'])
        
        if 'website' in api_summary['facebook'][0]:
            domain = get_domain(api_summary['facebook'][0]['website'])
            if domain is not None:
                api_domains.append(domain)
    
    if 'facebook_closed' not in rule_input:
        rule_input['facebook_closed'] = False
            
#     if len(api_summary.get('bbb',{})) > 0:
#         if '' in api_summary['bbb']:
#             rule_input[''] = api_summary['bbb'][]
            
    if len(api_summary.get('dataMerchant',{})) > 0:
        if 'bad_merchant' in api_summary['dataMerchant'][0]:
            rule_input['dmerch_hit'] = api_summary['dataMerchant'][0]['bad_merchant']
            
    if len(api_summary.get('sodaFoodInspection',{})) > 0:
        if 'closed' in api_summary['sodaFoodInspection'] and 'inactive' in api_summary['sodaFoodInspection']:
            if isInt(api_summary['sodaFoodInspection']['closed']) and isInt(api_summary['sodaFoodInspection']['inactive']):
                rule_input['doh_inactive'] = int(api_summary['sodaFoodInspection']['closed']) & int(api_summary['sodaFoodInspection']['inactive'])
            else:
                rule_input['doh_inactive'] = False
        else:
            rule_input['doh_inactive'] = False
    
    user_domain = get_domain(user_query.get('business',{}).get('website',None))
    if user_domain is not None:
        rule_input['domain'] = user_domain
    
    if len(api_domains) > 0:
        rule_input['api_domains'] = list(set(api_domains))
    if len(api_phones) > 0:
        rule_input['api_phones'] = list(set(api_phones))
        
    user_phone = user_query.get('business',{}).get('phone',None)
    if user_phone is not None:
        rule_input['phone'] = user_phone
    
    if 'doh_inactive' not in rule_input and 'sodaFoodInspection' in api_calls and 'sodaFoodInspection' not in api_summary.get('failed_apis',{}):
        rule_input['doh_inactive'] = False
            
    if len(api_summary.get('whoIs',{})) > 0:
        if 'whois_match' in api_summary['whoIs'][0]:
            rule_input['domain_valid'] = api_summary['whoIs'][0]['whois_match']
        
        if 'domain_created' in api_summary['whoIs'][0]:
            rule_input['whois_tenure'] = days_since(api_summary['whoIs'][0]['domain_created'])/365
        
        if 'domain_expiry' in api_summary['whoIs'][0]:
            rule_input['whois_expiration'] = days_since(api_summary['whoIs'][0]['domain_expiry']) * -1/365
        
    if valid_email != None and isinstance(valid_email,dict):
        rule_input['email_valid'] = True
        
    rule_input['fein_valid'] = validate_fein(user_query.get('business',{}).get('fein',''))
    
    if len(set(['duns_revenue','exp_revenue']) - set(rule_input.keys())) == 1: # if one of the item is missing
        if 'duns_revenue' not in rule_input:
            rule_input['duns_revenue'] = np.nan
        if 'exp_revenue' not in rule_input:
            rule_input['exp_revenue'] = np.nan
    
    if len(set(['bbb_tib','exp_yib']) - set(rule_input.keys())) == 1:
        if 'bbb_tib' not in rule_input:
            rule_input['bbb_tib'] = np.nan
        if 'exp_yib' not in rule_input:
            rule_input['exp_yib'] = np.nan
            
    if isInt(business.get('existing_mca_count',None)):
        rule_input['mca_cnt'] = business['existing_mca_count']
    
    return rule_input

@timed
def generate_scoring_input(user_query, api_summary):
    """Create input to run the scoring service
    Args:
        user_query: user input
        api_summary: summary of api_calls
    Returns:
        dict: input to scoring service
    """
    scoring_input = {}
    business_info = {}
    _now = datetime.now()
    if 'request_id' in user_query['header']:
        business_info['id'] = str(user_query['header']['request_id'])



    business_info['submission_date'] = "{:%Y-%m-%d}".format(_now)
    business_info['is_renewal'] = False
    business_info['is_winback'] = False
    business_info['intake_naics'] = user_query['business'].get('naics',None)
    business_info['intake_sic'] = user_query['business'].get('sic',None)
    business_info['state'] = user_query['business'].get('state',None)
    business_info['zip_code'] = user_query['business'].get('zip',None)
    
    business_info['business_start_date'] = "{:%Y-%m-%d}".format(datetime.strptime(user_query['business']['business_start_date'],'%m/%d/%Y'))
    if user_query['business'].get('ownership_start_date',None) == None:
        user_query['business']['ownership_start_date'] = user_query['business']['business_start_date']
    business_info['ownership_start_date'] = "{:%Y-%m-%d}".format(datetime.strptime(user_query['business']['ownership_start_date'],'%m/%d/%Y'))

    scoring_input['business_info'] = business_info
    ownership_info = []

    for consumer in user_query['consumers']:
        if not is_valid_ssn(consumer.get('ssn','')):
            continue
        owner = {}
        owner['id'] = str(consumer['id'])
        owner['name_first'] = consumer.get('name',{}).get('first','')
        owner['name_last'] = consumer.get('name',{}).get('first','')
        if 'credit_pull_consent' in consumer:
            owner['permissible_purpose'] = consumer['credit_pull_consent']
        else:
            owner['permissible_purpose'] = True
        owner['ssn_last_four'] = consumer.get('ssn','')[-4:]
        owner['percent_ownership'] = consumer['ownership_percentage']
        if 'signer_for_contracts' in consumer:
            owner['signer_for_contracts'] = consumer['signer_for_contracts']
        else:
            owner['signer_for_contracts'] = True
        if 'bank_info_responsible' in consumer:
            owner['bank_info_responsible'] = consumer['bank_info_responsible']
        else:
            owner['bank_info_responsible'] = True
        ownership_info.append(owner)

    scoring_input['owner_info'] = ownership_info
    apis_to_score = [ 'nfd', 'ln_crim', 'exp_csm', 'exp_ipv2', 'exp_sbcs', 'exp_ipv2bld', 'exp_sbcsbld', 'ln_sbrs']
    allowed_fileds = ['id','response_code','created_on','data','owner_id','num_records','nfd_value','timestamp']
    for api in api_summary:
        if api in apis_to_score:
            scoring_input_api = []
            for item in api_summary[api]:
                scoring_input_api_item = {}
                for key in item:
                    if key in allowed_fileds:
                        scoring_input_api_item[key] = item[key]
                if len(scoring_input_api_item) > 0:
                    scoring_input_api.append(scoring_input_api_item)
            
            if len(scoring_input_api) > 0:
                scoring_input[api] = scoring_input_api
        
    #scoring_input.update(api_summary)
    return scoring_input
@timed    
def generate_offers_input(user_query, api_calls, api_summary,scoring_results,conf):
    """Create input to run the scoring service
    Args:
        user_query: user input
        api_calls: result of all api calls
        api_summary: summary of api_calls
        scoring_results: output of scoring service
    Returns:
        dict: input to profitability service
    """
    offers_input = {}
    offers_input['renewal_ind'] = False
    offers_input['avg_deposit'] = float(user_query.get('business',{}).get('deposit_amt',0))
    offers_input['naics'] = user_query.get('business',{}).get('naics',None)
    offers_input['sic'] = user_query.get('business',{}).get('sic',None)
    offers_input['state'] = user_query.get('business',{}).get('state',None)
    offers_input['entity_age'] = days_since(user_query.get('business',{}).get('business_start_date',None),'%m/%d/%Y')
    offers_input['score'] = scale_raw_score(scoring_results.get('result',{}).get('score_raw',0.999),conf)
    offers_input['offer_dt'] = "{:%Y-%m-%dT%H:%M:%S}".format(datetime.now())
    return offers_input   

def _format_summary(input_summary):
    """Formats 
    Args:
        input_summary: summary of api_calls
    Returns:
        dict: formatted summary for readability and understanding
    """
    output_summary = {}
    fields_to_pop = ['id','response_code','created_on','data','owner_id','timestamp','phones','website']
    acronym_fields = ['fico','cbo','naics','sic']
    apis_to_map = {'exp_csm':'experianConsumer',
                   'nfd':'experianNfd',
                   'ln_crim':'lexisNexisCriminal',
                   'factor_trust':'factorTrust',
                   'exp_ipv2':'experianIpv2',
                   'exp_sbcs':'experianSbcs',
                   'exp_ipv2bld':'experianIpv2Blended',
                   'exp_sbcsbld':'experianSbcsBlended',
                   'ln_sbrs':'lexisNexisSbrs'}
    fields_to_rename = {'nfd_value':'NFDIndex'}
    if 'pacerBk' in input_summary:
        input_summary.pop('pacerBk')
    for api in input_summary:
        if api == 'pacerEpnBk':
            if isinstance(input_summary[api][0],list):
                for sum_item in input_summary[api][0]:
                    if isinstance(sum_item,dict) and 'owner_id' in sum_item:
                        sum_item.pop('owner_id')
            
        if api != 'failed_apis' and api != 'skipped_apis' and len(input_summary[api]) > 0:
            outs = []
            for item in input_summary[api]:
                if not isinstance(item,dict):
                    outs.append(item) # assuming these are handled already
                    continue
                out = dict(item)
                for field in fields_to_pop:
                    if field in out:
                        out.pop(field)
                for field in acronym_fields:
                    if field in out:
                        out[field.upper()] = out.pop(field)
                for field in fields_to_rename:
                    if field in out:
                        out[fields_to_rename[field]] = out.pop(field)
                        
                if len(out) > 0:
                    outs.append(out)
            if len(outs) > 0:
                out = outs
                if len(outs) == 1:
                    out = outs[0]
                if api in apis_to_map:
                    output_summary[apis_to_map[api]] = out
                else:
                    output_summary[api] = out
        else:
            output_summary[api] = input_summary[api]
    return output_summary

def _round_down(num, divisor=1):
    """Rounds down the given number
    args:
        num: number to round down
        divisor: factor to round down. Example: 1,10,100 etc.
    Retunrs:
        rounded down value.
    Example: _round_down(1428.33,100) returns 1400.0
    """
    if num is None or divisor is None:
        return num
    return num - (num%divisor)


def _get_offers_pricing(pricing, offers, scoring_results, conf, user_access):
    """Estimate the factor for each offer and term
    args:
        offers: output from profitability(offer calculation)
        scoring_results: output from scoring service 
    Retunrs:
        dict: factor for each rtr/term
    """
    offer_pricing = []
    if 'line_out' in offers and len(offers['line_out']) > 0 and 'result' in scoring_results and 'score' in scoring_results['result'] and isInt(scoring_results['result']['score']):
        
        # get scoring ranges from config files
        score_ranges = {}
        for col in pricing.columns:
            score_range = col.split('_')
            if len(score_range) == 2 and isInt(score_range[0]) and isInt(score_range[1]):
                score_ranges[col] = (int(score_range[0]), int(score_range[1]))
        score = int(scoring_results['result']['score'])
        
        score_range = None
        
        for item in score_ranges:
            if score >= score_ranges[item][0] and score <= score_ranges[item][1]:
                score_range = item
                break
        
        if score_range != None:
            commission_factor = conf.get('commission_factor',{}).get('value',.12)
            if isinstance(user_access, dict) and isNumber(user_access.get('commission_factor',None)):
                commission_factor = user_access.get('commission_factor')
            for term in offers['line_out']:
                pricing_item = {'rtr':_round_down(offers['line_out'][term],100)}
                rtr = pricing[pricing['Term'] == term][score_range]
                if len(rtr) > 0:
                    factor = float(list(rtr)[0]) + float(commission_factor)
                    pricing_item['factor'] = min(factor, float(conf.get('commission_cap',{}).get('value','1.49')))
                    pricing_item['term'] = term
                    pricing_item['advance'] = _round_down(offers['line_out'][term]/factor,100)
                    offer_pricing.append(pricing_item)
    else:
        for term in offers['line_out']:
            if int(term) >= 3 and int(term) <= 15:
                pricing_item = {'term':term,'rtr':_round_down(offers['line_out'][term],100)}
                offer_pricing.append(pricing_item)
    return offer_pricing

def _get_decision(rules_result, scoring_results, offers,pricing, conf, user_access):
    """Estimate the factor for each offer and term
    args:
        rules_result: output from run_rules
        offers: output from profitability(offer calculation)
        scoring_results: output from scoring service 
    Retunrs:
        dict: with node for summary of offer, score and rule
    """    
    decision = {}#{'warnings':[],'offers':offers,'decline_reasons':[]}
    if rules_result != None and 'flag' in rules_result:
        warnings = []
        warning_rule = []
        for item in rules_result['flag']:
            if isinstance(item, list) and len (item) > 0:
                warning_rule.append(item.pop())
                warnings.append(item.pop(0))
        decision['warnings'] = list(set(warnings))
        rules_result['flag'] = warning_rule
    if rules_result != None and 'decline' in rules_result:
        decline_reasons = []
        decline_rule = []
        for item in rules_result['decline']:
            if isinstance(item, list) and len (item) > 0:
                decline_rule.append(item.pop())
                decline_reasons.append(item.pop(0))
        decision['decline_reasons'] = list(set(decline_reasons))
        decision['decision'] = 'Not prequalified'
        decision['message'] = conf.get('decision_text',{}).get('Not prequalified','No Instant Prequalified offers are available based upon the information provided.  You may submit a full Prequalification Request for this merchant through the standard channels.')
        rules_result['decline'] = decline_rule
        
    if scoring_results != None and 'result' in scoring_results:
        score = math.ceil(scale_raw_score(scoring_results['result'].get('score_raw',.999),conf)* 1000) 
        decision['scores'] = {'score':score, 'score_adj':score} # both score and adj score are same for now.
    
    if offers != None and 'line_out' in offers:
        decision['offers'] = _get_offers_pricing(pricing, offers, scoring_results, conf, user_access)
        decision['decision'] = 'Prequalified'
    
    return decision


def _format_rule_input(rule_input, api_summary, all_api_responses):
    if len(set(['duns_revenue','exp_revenue','bbb_tib','exp_yib']) & set(rule_input.keys())) > 0:
        if 'duns_revenue' in rule_input and (rule_input['duns_revenue'] is np.nan or rule_input['duns_revenue'] is None):
            rule_input.pop('duns_revenue')
        if 'exp_revenue' in rule_input and (rule_input['exp_revenue'] is np.nan or rule_input['exp_revenue'] is None):
            rule_input.pop('exp_revenue')
            
        if 'bbb_tib' in rule_input and (rule_input['bbb_tib'] is np.nan or rule_input['bbb_tib'] is None):
            rule_input.pop('bbb_tib')
        if 'exp_yib' in rule_input and (rule_input['exp_yib'] is np.nan or rule_input['exp_yib'] is None):
            rule_input.pop('exp_yib')
    
    if 'doh_inactive' in rule_input and 'sodaFoodInspection' in all_api_responses and 'sodaFoodInspection' not in api_summary.get('failed_apis',{}) and ('sodaFoodInspection' not in api_summary or len(api_summary['sodaFoodInspection']) == 0) :
        rule_input.pop('doh_inactive')
    
    if 'bbb' in all_api_responses and 'bbb' not in api_summary.get('failed_apis',{}) and ('bbb' not in api_summary or len(api_summary['bbb']) == 0):
        if 'bbb_inactive' in rule_input:
            rule_input.pop('bbb_inactive')
        
        if 'bbb_rating' in rule_input:
            rule_input.pop('bbb_rating')
    
    if 'yelp_closed' in rule_input and 'yelp' in all_api_responses and 'yelp' not in api_summary.get('failed_apis',{}) and ('yelp' not in api_summary or len(api_summary['yelp']) == 0 or 'permanently_closed' not in api_summary['yelp']) :
        rule_input.pop('yelp_closed')
    if 'google_closed' in rule_input and 'google' in all_api_responses and 'google' not in api_summary.get('failed_apis',{}) and ('google' not in api_summary or len(api_summary['google']) == 0 or 'permanently_closed' not in api_summary['google']) :
        rule_input.pop('google_closed')
    
    if 'facebook_closed' in rule_input and 'facebook' in all_api_responses and 'facebook' not in api_summary.get('failed_apis',{}) and ('facebook' not in api_summary or len(api_summary['facebook']) == 0 or 'permanently_closed' not in api_summary['facebook']) :
        rule_input.pop('facebook_closed')
    
    return rule_input

def items_same(keys, item1, item2):
    """Checks if two dict has same value for given keys
    args:
        keys: list of keys to check
        item1: dict 1 for comparison
        item2: dict 2 for comparison
    Retunrs:
        bool: True if all value matches - otherwise False
    """
    for key in keys:
        if key not in item1 and key not in item2:
            continue
        if key not in item1 or key not in item2 or str(item1[key]) != str(item2[key]):
            return False
    return True

@timed
def run_score(scoring_input):
    """wrapper function to call scoring service. 
    Args:
        scoring_input: input dict to scoring service
    Returns:
        json result of scoring service
    """    
    return run_end2end(json.dumps(scoring_input))

@timed
def run_offer(offer_input):
    """wrapper function to call profitability service. 
    Args:
        offer_input: input dict to offer generation service
    Returns:
        json result of profitability service
    """ 
    return run_knn(offer_input,load_knn(None),['renewal_ind','score','avg_deposit','naics','entity_age','state','offer_dt','bankstmt_months','sic'])

def _get_consumers_for_l2c(_api_summary, apis,consumers,conf,score):
    """ function to determine if TU L2C api call is required or not
    Use average FICO and score to determine eligibility
    """
    consumers_for_pid = []
    try:
        if not isinstance(_api_summary,dict):
            return consumers_for_pid

        owner_ids_pid_done = []
        for pid_summary in _api_summary.get('l2c',[{}]):
            if isinstance(pid_summary,dict) and 'owner_id' in pid_summary:
                owner_ids_pid_done.append(pid_summary['owner_id'])

        fico_cut_off = 600 # default value
        score_cut_off = 160 # default value

        if 'cutoff_for_l2c' in conf and isNumber(conf['cutoff_for_l2c'].get('fico',None)) and isNumber(conf['cutoff_for_l2c'].get('score',None)): # pick from database if present
            fico_cut_off = float(conf['cutoff_for_l2c'].get('fico',None))
            score_cut_off = float(conf['cutoff_for_l2c'].get('score',None))
        if score <= score_cut_off:
            return consumers_for_pid

        ficos = []

        if isinstance(_api_summary.get('exp_csm',None),list):# to handle multiple cust scenario
            for csm in _api_summary.get('exp_csm'):
                if isinstance(csm, dict) and isNumber(csm.get('fico',None)) and\
                float(csm.get('fico',999)) < fico_cut_off and\
                float(csm.get('fico',999)) <=900 and\
                float(csm.get('fico',0)) > 350:
                    ficos.append(csm.get('fico'))

            if np.mean(ficos) < fico_cut_off and 'l2c' in apis:
                    for consumer in consumers:
                        if isinstance(consumer,dict) and consumer.get('id','') not in owner_ids_pid_done and consumer not in consumers_for_pid:
                            consumers_for_pid.append(consumer)
    except Exception as e:
        pass
    return consumers_for_pid

def _get_consumers_for_pid(_api_summary, apis,consumers,conf):
    """ function to determine if experian PID/FPD api call is required or not
    """
    consumers_for_pid = []
    if not isinstance(_api_summary,dict):
        return consumers_for_pid
    
    owner_ids_pid_done = []
    for pid_summary in _api_summary.get('experianPid',[{}]):
        if isinstance(pid_summary,dict) and 'owner_id' in pid_summary:
            owner_ids_pid_done.append(pid_summary['owner_id'])
    
    cut_off = 399
    
    if 'cbo_cutoff_for_pid' in conf and isNumber(conf['cbo_cutoff_for_pid']): # to handle sole scenario
        cut_off = float(conf['cbo_cutoff_for_pid'])
    if isinstance(_api_summary.get('exp_csm',None),list):# to handle multiple cust scenario
        for csm in _api_summary.get('exp_csm'):
            if isinstance(csm, dict) and isNumber(csm.get('cbo',0)) and float(csm.get('cbo',0)) > cut_off and float(csm.get('cbo',0)) <=1000 and 'experianpid' in apis:
                for consumer in consumers:
                    if isinstance(consumer,dict) and consumer.get('id','') not in owner_ids_pid_done and consumer not in consumers_for_pid:
                        consumers_for_pid.append(consumer)
    return consumers_for_pid

def _get_consumers_for_pacer_bk(_api_summary, consumers):
    """ function to determine if pacer api is required. If required, return a list of consumer to call pacer.
    """
    consumers_for_pacer_bk = []
    if not isinstance(_api_summary,dict):
        return consumers_for_pacer_bk
    
    owner_ids_pacer_done = []
    for bk_case_summary in _api_summary.get('pacerBk',[[]]):
        if isinstance(bk_case_summary,dict) and 'owner_id' in bk_case_summary:
            owner_ids_pacer_done.extend(bk_case_summary['owner_id'])
    for consumer_summary in _api_summary.get('exp_csm',[]):
        if isinstance(consumer_summary,dict) and 'OpenBankruptcies' in consumer_summary and len(consumer_summary['OpenBankruptcies']) > 0 and 'owner_id' in consumer_summary:
            for consumer in consumers:
                if isinstance(consumer,dict) and consumer.get('id',None) == consumer_summary.get('owner_id','') and consumer_summary.get('owner_id','') not in owner_ids_pacer_done:
                    if consumer not in consumers_for_pacer_bk:
                        consumers_for_pacer_bk.append(consumer)
    
        
    
    return consumers_for_pacer_bk


def call_and_update_api(apis_to_call,user_query,apis,cached_apis,called_api_responses,all_api_responses,api_summary,failed_apis):        
    """
    wrapper function to call a set of apis and update summary and results
    """
    _apis_to_call = []
    for api in apis_to_call:
        if api in apis and api not in cached_apis:
            _apis_to_call.append(api)
    api_responses = call_apis(user_query, _apis_to_call) 
    called_api_responses.update(api_responses)
    all_api_responses.update(called_api_responses)
    api_summary.update(get_api_summary(user_query,api_responses))
    failed_apis.update(api_summary.get('failed_apis',{}))
    return called_api_responses, all_api_responses, api_summary, failed_apis


status = {'started':0,'api_calls_complete':1,'scored':2,'offers_generated':3,'duplicate_query':4, 'error':99}
@timed    
def run_preapproval(user_query, apis,request_id, _id,conf,user_id,input_validation,user_access,cached_objects,update=False):
    """ Executes preapproval
    Args:
        user_query: dict - business and consumers info
        apis - list of apis to call
        _id - unique preapproval identifier
    Returns:
        dict - preapproval result
    """ 
    user_query['apis'] = apis
    _business,_consumers = create_session(user_query,_id,user_id,conf,update)
    user_query.pop('apis')
    pa_status = status['started']
    user_query['business']['id'] = str(_business['id'])
    user_query['business']['recent_api_query_id'] = _business['recent_api_query_id']
    consumer_ids = []
    for ix in range(len(user_query['consumers'])):
        if 'id' in _consumers[ix]:
            user_query['consumers'][ix]['id'] = str(_consumers[ix]['id'])
            user_query['consumers'][ix]['recent_api_query_id'] = _consumers[ix]['recent_api_query_id']
            consumer_ids.append(user_query['consumers'][ix]['id'])
    user_query['header'] = {'request_id':_id}
    db_info = get_db_info(user_query,conf,cached_objects)
    recent_queries  = db_info.get('queries_cached_apis',{}).get('recent_queries',{})
    recent_apis = db_info.get('queries_cached_apis',{}).get('recent_apis',{})
    api_query_ids = db_info.get('queries_cached_apis',{}).get('api_query_ids',{})    
    apis_to_call_by_consumer = db_info.get('queries_cached_apis',{}).get('apis_to_call_by_consumer',{})
    #authentication = db_info.get('authentication',{})
    user_query['conf'] = conf
    apis = [x.lower() for x in apis]
    all_api_responses = {}
    called_api_responses = {}
    api_summary = {}
    apis_to_call = []
    scoring_results = None
    scoring_result_json = None
    scoring_input = None
    if isinstance(recent_apis,dict):
        
        apis_to_pop = []
        for recent_api in recent_apis:
            if recent_api.lower() not in apis:
                apis_to_pop.append(recent_api)
        for api in apis_to_pop:
            recent_apis.pop(api) # remove apis from cache that are not in request
    if recent_queries is not None and len(recent_queries) > 0:
        for req_id in recent_queries:
            if str(req_id) == str(request_id):
                continue# Ignore current request for historic check
            recent_consumers = recent_queries[req_id]['consumers']
            if not isinstance(recent_consumers,str):
                continue
            recent_consumers = recent_consumers.split(',')
            recent_consumers = [c.strip() for c in recent_consumers]
            consumers_match = False
            if  set(consumer_ids) == set(recent_consumers):
                consumers_match = True

            recent_request = recent_queries[req_id]['request']
            resp_id = recent_queries[req_id]['response_id']
            if 'apis' not in recent_request:
                recent_request['apis'] = conf.get('apis',[])
            apis_match = len(set([ api.lower() for api in apis]) & set([ api.lower() for api in recent_request.get('apis',[])])) == len(set([api.lower() for api in apis]))
            apis_not_expired = len(set([ api.lower() for api in apis]) & set([ api.lower() for api in recent_apis.keys()])) == len(set([api.lower() for api in apis]))
            apis_recent = True
            for api in api_query_ids:
                # check if the api call happened after the query. If the api was not recent we need to redo scoring and rules
                if (api in apis or api.lower() in apis) and isInt(api_query_ids[api]) and isInt(req_id) and int(api_query_ids[api]) > int(req_id):
                    apis_recent = False
                    break
                    
            if apis_match and apis_not_expired and apis_recent and consumers_match and items_same(['business_start_date', 'ownership_start_date', 'email', 'website', 'phone', 'revenue', 'deposit_amt', 'naics', 'sic'],recent_request.get('business',{}),user_query['business']):
                # return a response from db when matching consumers, all apis are recent and query is duplicate
                response = get_full_response_by_id(resp_id, fk=conf['fk'])
                if 'decisions' in response:
                    pa_status = status['duplicate_query']
                    if update:
                        response['response_id'] = _id
                        if _id != resp_id:
                            response['actual_response_id'] = resp_id
                    else:
                        response['response_id'] = resp_id
                    
                    response['status'] = pa_status
                    response['code'] = 200
                    response['inputs'] = input_validation
                    update_response(request_id, response['response_id'], _business, _consumers, response, called_api_responses,pa_status,conf['fk'])
                    return transform_response_for_user(response, user_access,conf)
    if 'whoIs' in recent_apis and user_query['business'].get('website',None) != None and recent_apis['whoIs'].get('raw_data',[{}]) != None and len(recent_apis['whoIs'].get('raw_data',[{}])) > 0 and recent_apis['whoIs'].get('raw_data',[{}])[0].get('request',{}).get('domain','').lower() not in user_query['business'].get('website','').lower():
        # force who is api call again if websites don't match
        recent_apis.pop('whoIs')
    
    # populate deposit amt to run rules
    if ('deposit_amt' not in user_query['business'] or not isNumber(user_query['business']['deposit_amt'])) and 'revenue' in user_query['business'] and isNumber(user_query['business']['revenue']):
        try:
            user_query['business']['deposit_amt'] = float(user_query['business']['revenue'])/12
        except:
            'Do nothing'
    
    cached_apis = [api.strip().lower() for api in recent_apis.keys()]
    for api in recent_apis:
        if api.lower() in apis:
            all_api_responses[api] = recent_apis[api]
    valid_email = None
    if user_query.get('business',{}).get('email',None) != None:
        try:
            valid_email = validate_email_address(user_query['business']['email'])
            
        except Exception as e:            
            valid_email = False
            pass
    # run rules on user supplied info - ignore api summary from history to avoid running rules with partial info
    result = run_rules(generate_rule_input(user_query, {}, {}, scoring_results,db_info,conf,update, valid_email),db_info['rules'])
    api_summary = get_api_summary(user_query,all_api_responses)
    
    failed_apis = {}
    if 'decline' not in result:
        initial_apis = ['datamerchant'] # non dependent external apis - starting with low cost and high predictive
        
        called_api_responses, all_api_responses, api_summary, failed_apis = call_and_update_api(initial_apis,user_query,
                                                                                        apis,
                                                                                        cached_apis,
                                                                                        called_api_responses,
                                                                                        all_api_responses,
                                                                                        api_summary,failed_apis)
    # run the rules
    result = run_rules(generate_rule_input(user_query, all_api_responses, api_summary,scoring_results,db_info,conf,update,valid_email),db_info['rules'])
    
    
    
    crowd_sourced_apis = ['google','yelp','facebook','sodafoodinspection','bbb']
    if 'website' in user_query['business'] and isinstance(user_query['business']['website'],str) and len(user_query['business']['website']) > 0:
         crowd_sourced_apis.append('whois')
    
    if 'decline' not in result:
        # setting cached object to user_query for sodafood - DONT use any where else
        t_user_query = dict(user_query)
        t_user_query['doh_cache_data'] = {'doh_states':cached_objects['doh_states'],
                                        'dba_vectorizer':cached_objects['dba_vectorizer'],
                                       'name_vectorizer':['name_vectorizer']}
        # calling crowd sourced/opej source apis
        called_api_responses, all_api_responses, api_summary, failed_apis = call_and_update_api(crowd_sourced_apis,t_user_query,
                                                                                        apis,
                                                                                        cached_apis,
                                                                                        called_api_responses,
                                                                                        all_api_responses,
                                                                                        api_summary,failed_apis)
        del t_user_query # clearing it from memory
    # rerun the rules
    result = run_rules(generate_rule_input(user_query, all_api_responses, api_summary,scoring_results,db_info,conf,update,valid_email),db_info['rules'])
    
    if 'decline' not in result:
        _apis = ['experianconsumer'] # experian api
        
        called_api_responses, all_api_responses, api_summary, failed_apis = call_and_update_api(_apis,user_query,
                                                                                        apis,
                                                                                        cached_apis,
                                                                                        called_api_responses,
                                                                                        all_api_responses,
                                                                                        api_summary,failed_apis)
        
        
        # run the rules - consumer
        result = run_rules(generate_rule_input(user_query, all_api_responses, api_summary,scoring_results,db_info,conf,update,valid_email),db_info['rules'])
    
    if 'decline' not in result :
        _apis = ['experiannfd'] # experian NFD
        
        called_api_responses, all_api_responses, api_summary, failed_apis = call_and_update_api(_apis,user_query,
                                                                                        apis,
                                                                                        cached_apis,
                                                                                        called_api_responses,
                                                                                        all_api_responses,
                                                                                        api_summary,failed_apis) 
    
        # rerun the rules - for nfd
        result = run_rules(generate_rule_input(user_query, all_api_responses, api_summary,scoring_results,db_info,conf,update,valid_email),db_info['rules'])
    
    # Doing PID and Pacer as these are trigger/condition based    
    if 'decline' not in result:
        apis_to_call = []
        
        recent_apis_lower = {}        
        recent_apis.update(recent_apis_lower)
        for api_consumer_ids in apis_to_call_by_consumer:
            api_consumers = []
            for cid in api_consumer_ids:
                for consumer in user_query['consumers']:
                    if consumer.get('id',None) == cid:
                        api_consumers.append(consumer)
            consumer_ids_apis_to_call = apis_to_call_by_consumer[api_consumer_ids]
            
            apis_to_call = []
            for api in consumer_ids_apis_to_call:
                if api.lower() in apis:
                    if api.lower() == 'pacerbk' or api.lower() == 'experianpid':
                        continue  #pacer is handled using _get_consumers_for_pacer_bk - need only for experian bk
                    else:    
                        apis_to_call.append(api)
            q = dict(user_query)
            q['consumers'] = api_consumers
            called_api_responses.update(call_apis(q, apis_to_call))
                        
            
            _keys = {i.lower():i for i in recent_apis.keys()}
            c_keys = {i.lower():i for i in called_api_responses.keys()}
            for api in apis_to_call:
                hist_resp = recent_apis.get(_keys.get(api.lower(),''),{}).get('raw_data',{})
                new_resp = called_api_responses.get(c_keys.get(api.lower(),''),{}).get('raw_data',{})
                combined_resp = []
                
                for c_id in consumer_ids:
                    if len(hist_resp) > 0:
                        for item in hist_resp:
                            if item != None and c_id == item['consumer_id']:
                                combined_resp.append(item)
                    if len(new_resp) > 0:
                        for item in new_resp:
                            if item != None and c_id == item['consumer_id']:
                                combined_resp.append(item)
                if api.lower() in c_keys:
                    all_api_responses[c_keys[api.lower()]] = dict(called_api_responses[c_keys[api.lower()]])
                    all_api_responses[c_keys[api.lower()]]['raw_data'] = combined_resp
        api_summary = get_api_summary(user_query,all_api_responses)
        failed_apis.update(api_summary.get('failed_apis',{}))
        pacer_consumers = _get_consumers_for_pacer_bk(api_summary, user_query['consumers'])
        pid_consumers = _get_consumers_for_pid(api_summary, apis,user_query['consumers'],conf)
        
        if len(pacer_consumers) > 0 and 'pacerbk' in apis:
            q = dict(user_query)
            q['consumers'] = pacer_consumers
            if 'business' in q:
                q.pop('business')
            called_api_responses.update(call_apis(q, ['pacerBk']))
            if isinstance(all_api_responses.get('pacerBk',{}).get('raw_data',None),list):
                new_raw_data = called_api_responses.get('pacerBk',{}).get('raw_data',[])
                if len(new_raw_data) > 0:
                    all_api_responses['pacerBk']['raw_data'].extend(new_raw_data)
                    all_api_responses['pacerBk'].update(called_api_responses.get('pacerBk',{}))
            else:
                all_api_responses.update(called_api_responses)
            api_summary = get_api_summary(user_query,all_api_responses)
            failed_apis.update(api_summary.get('failed_apis',{}))
        
        if len(pid_consumers) > 0 and 'experianpid' in apis:
            q = dict(user_query)
            q['consumers'] = pid_consumers
            called_api_responses.update(call_apis(q, ['experianpid']))
            
            if isinstance(all_api_responses.get('experianPid',{}).get('raw_data',None),list):
                new_raw_data = called_api_responses.get('experianPid',{}).get('raw_data',[])
                if len(new_raw_data) > 0:
                    all_api_responses['experianPid']['raw_data'].extend(new_raw_data)
                    all_api_responses['experianPid'].update(called_api_responses.get('experianPid',{}))
            else:
                all_api_responses.update(called_api_responses)
            api_summary = get_api_summary(user_query,all_api_responses)
            failed_apis.update(api_summary.get('failed_apis',{}))
    
    if 'whois' not in api_summary:
        if 'google' in api_summary and 'website' in api_summary['google']:
            user_query['business']['website'] = api_summary['google']['website']
        elif 'facebook' in api_summary and 'website' in api_summary['facebook']:
            user_query['business']['website'] = api_summary['facebook']['website']
        elif 'bbb' in api_summary and 'website' in api_summary['bbb']:
            user_query['business']['website'] = api_summary['bbb']['website']
    
    # run the rules
    result = run_rules(generate_rule_input(user_query, all_api_responses, api_summary,scoring_results,db_info,conf,update,valid_email),db_info['rules']) 
    
        
    if 'decline' not in result:
        api_responses = {}
        apis_to_call = []
        
        if 'experianipv2' in apis and 'experianipv2' not in cached_apis:
            apis_to_call.append('experianipv2')
        
        
        api_responses = call_apis(user_query, apis_to_call) # call ipv2 to get filenumber
        called_api_responses.update(api_responses)
        all_api_responses.update(called_api_responses)
        api_summary.update(get_api_summary(user_query,api_responses))
        failed_apis.update(api_summary.get('failed_apis',{}))  
        
        if len(failed_apis) > 0:
            if 'experianIpv2' in failed_apis:
                if 'experianConsumer' in failed_apis:
                    err_resp = {'status':status['error'],'message':'Error occurred during Experian api call. Contact support with UID: ' + str(_id),'response_id':_id,'apis':{'data':all_api_responses,'summary':_format_summary(api_summary)},'code':500,'inputs':input_validation}
                    if len(user_query['consumers']) == 1:
                        update_response(request_id, _id, _business, _consumers, err_resp, called_api_responses,status['error'],conf['fk'])
                        raise ExperianFailure()
                    else:
                        raise_error= True
                        for csm_resp in all_api_responses.get('experianConsumer',{}).get('raw_data'):
                            if 'status' in csm_resp and csm_resp['status'] == 'OK':
                                raise_error = False
                                break
                        if raise_error: # raise error when consumer failed for all consumers
                            update_response(request_id, _id, _business, _consumers, err_resp, called_api_responses,status['error'],conf['fk'])
                            raise ExperianFailure()
        
        api_responses = {}
        apis_to_call = []
        ipv2_multi_matches = []
        if 'experianipv2' in apis and 'exp_ipv2' in api_summary and len(api_summary['exp_ipv2']) > 0 and api_summary['exp_ipv2'][0].get('repull_with_file_number', False):
            apis_to_call.append('experianipv2')
            ipv2_multi_matches = called_api_responses.get('experianIpv2',{}).get('raw_data',[])
        if 'experiansbcs' in apis and 'experiansbcs' not in cached_apis:
            apis_to_call.append('experianSbcs')    
        if 'experianipv2blended' in apis and 'experianipv2blended' not in cached_apis:
            apis_to_call.append('experianIpv2Blended')
        if 'experiansbcsblended' in apis and 'experiansbcsblended' not in cached_apis:
            apis_to_call.append('experianSbcsBlended')
        
        if len(apis_to_call) > 0:
            business = {}
            if 'exp_ipv2' in api_summary and len(api_summary['exp_ipv2']) > 0 and 'ExperianFileNumber' in api_summary['exp_ipv2'][0]:
                business['file_number'] = api_summary['exp_ipv2'][0]['ExperianFileNumber']
                query = {'consumers':user_query['consumers'],'business':business,'conf':conf,'header':user_query['header']}
                api_responses = call_apis(query, apis_to_call) # call business and blended apis by filenumber
                called_api_responses.update(api_responses)
                all_api_responses.update(called_api_responses)
                if len(ipv2_multi_matches) > 0:
                    all_api_responses['experianIpv2']['raw_data'].extend(ipv2_multi_matches)
                api_summary.update(get_api_summary(user_query,api_responses))
                failed_apis.update(api_summary.get('failed_apis',{}))
            else:
                failed_apis = api_summary.get('failed_apis',{})
                for api in apis_to_call:
                    if api.lower() in apis:
                        apis.remove(api.lower()) # don't call these without file number
                        called_api_responses.update({api:{'status':'Request not sent - IPV2 not complete'}})
                        #failed_apis[api] = 'Error'
                all_api_responses.update(called_api_responses)
                api_summary['failed_apis'] = failed_apis   
        
        # populate some data from api summaries
        try:
            if ('deposit_amt' not in user_query['business'] or not isNumber(user_query['business']['deposit_amt']) or int(user_query['business']['deposit_amt']) == 0) and 'exp_sbcs' in api_summary and 'sales_revenue' in api_summary['exp_sbcs'][0] and int(api_summary['exp_sbcs'][0]['SalesRevenue']) > 0:
                user_query['business']['revenue'] = api_summary['exp_sbcs'][0]['SalesRevenue']
                user_query['business']['deposit_amt'] = float(user_query['business']['revenue'])/12
        except:
            'Do nothing'
        
        experian_has_industry = False
        industry_changed = False
        # Use info from experian - if different/missing
        if api_summary != None and 'exp_sbcs' in api_summary:
            
            # Experian SIC/NAICS takes higer precendence
            if len(api_summary['exp_sbcs']) > 0 and ('naics' in api_summary['exp_sbcs'][0] or 'sic' in api_summary['exp_sbcs'][0]):
                experian_has_industry = True
            if 'naics' in api_summary['exp_sbcs'][0] and isInt(api_summary['exp_sbcs'][0]['naics']) and int(api_summary['exp_sbcs'][0]['naics']) != int(user_query.get('business',{}).get('naics',0)):
                user_query['business']['naics'] = api_summary['exp_sbcs'][0]['naics']
                industry_changed = True
            if 'sic' in api_summary['exp_sbcs'][0] and isInt(api_summary['exp_sbcs'][0]['sic']) and int(api_summary['exp_sbcs'][0]['sic']) != int(user_query.get('business',{}).get('sic',0)):
                user_query['business']['sic'] = api_summary['exp_sbcs'][0]['sic']
                industry_changed = True
            
            # Use experian info if user info is missing
            if 'years_in_business' in api_summary['exp_sbcs'][0]:
                user_query['business']['business_start_date'] = "{:%m/%d/%Y}".format(date_from_years(api_summary['exp_sbcs']['years_in_business']))
        
        
        if not experian_has_industry:
            api_responses = {}
            apis_to_call = []
            if 'd&bdirect' not in cached_apis and 'd&bdirect' in apis:
                apis_to_call.append('d&bdirect')
                dandb_user_query = dict(user_query)
                if 'exp_ipv2' in api_summary and len(api_summary['exp_ipv2']) > 0 and 'ExperianBusinessName' in api_summary['exp_ipv2'][0]:
                    # user experian business name if present
                    dandb_user_query['business']['dba'] = api_summary['exp_ipv2'][0]['ExperianBusinessName']
                api_responses = call_apis(dandb_user_query, apis_to_call) # call d&b to get sic number
                called_api_responses.update(api_responses)
                all_api_responses.update(called_api_responses)
                api_summary.update(get_api_summary(user_query,api_responses))
                failed_apis.update(api_summary.get('failed_apis',{}))
            if 'd&bdirect' in api_summary and len(api_summary['d&bdirect']) > 0 and 'sic' in api_summary['d&bdirect'][0] and int(api_summary['d&bdirect'][0]['sic']) != int(user_query.get('business',{}).get('sic',0)):
                    user_query['business']['sic'] = api_summary['d&bdirect']['sic']
                    industry_changed = True
        
        if industry_changed:
            db_info['high_risk_industries'] = get_high_risk_industries(cached_objects['df_risky_industry'], naics=user_query['business']['naics'],sic=user_query['business']['sic'])
        
            
        pa_status = status['api_calls_complete']
        scoring_input = generate_scoring_input(user_query, api_summary)
        if len(scoring_input) > 0:
            try:
                scoring_result_json = run_score(scoring_input)
                scoring_results = json.loads(scoring_result_json)
            except Exception as e:
                raise e
                ''#scoring_results = {}
            pa_status = status['scored']
    
    api_summary = get_api_summary(user_query,all_api_responses)
    rule_input = generate_rule_input(user_query, all_api_responses, api_summary, scoring_results,db_info,conf,update,valid_email)
    rules_result = run_rules(rule_input,db_info['rules'])
    # call other apis after scoring to save time/money
    if 'decline' not in rules_result:
        
        if scoring_results != None and 'result' in scoring_results and 'score' in scoring_results['result'] and isInt(scoring_results['result']['score']) and scoring_results['result']['score'] < 999:
            l2c_consumers = _get_consumers_for_l2c(api_summary, apis,user_query['consumers'],conf,scoring_results['result']['score'])
            if len(l2c_consumers) > 0:
                l2c_query = dict(user_query)
                l2c_query['consumers'] = l2c_consumers
                called_api_responses, all_api_responses, api_summary, failed_apis = call_and_update_api(['l2c'],l2c_query,
                                                                                        apis,
                                                                                        cached_apis,
                                                                                        called_api_responses,
                                                                                        all_api_responses,
                                                                                        api_summary,failed_apis) 
            
        called_apis = [api.lower() for api in all_api_responses.keys()]
        apis_to_call = list(set(apis) - (set(called_apis) | set(['experianpid','d&bdirect','pacerbk','l2c']))) # include conditional apis to exclude
        called_api_responses, all_api_responses, api_summary, failed_apis = call_and_update_api(apis_to_call,user_query,
                                                                                        apis,
                                                                                        cached_apis,
                                                                                        called_api_responses,
                                                                                        all_api_responses,
                                                                                        api_summary,failed_apis) 
        
    rule_input = generate_rule_input(user_query, all_api_responses, api_summary, scoring_results,db_info,conf,update,valid_email)
    rules_result = run_rules(rule_input,db_info['rules'])
    offers = None
    offer_input = None
    
    
    
    if 'decline' not in rules_result and scoring_results != None and 'result' in scoring_results and 'score' in scoring_results['result'] and isInt(scoring_results['result']['score']) and scoring_results['result']['score'] < 999: # check for decline
        offer_input = generate_offers_input(user_query, all_api_responses, api_summary,scoring_results,conf)
        if len(offer_input) > 0:
            offers = run_offer(dict(offer_input)) # sending a copy of input as the offer service is renaming them
            pa_status = status['offers_generated']
    decisions = _get_decision(rules_result, scoring_results, offers, db_info['pricing'], conf,user_access)
    if len(failed_apis) > 0:
        api_summary['failed_apis'] = failed_apis
    skipped_apis = []
    for api in all_api_responses:
        if 'raw_data' not in all_api_responses[api] or all_api_responses[api]['raw_data'] is None or len(all_api_responses[api]['raw_data']) == 0:
            skipped_apis.append(api)
    api_summary['skipped_apis'] = skipped_apis
    rule_input = _format_rule_input(rule_input, api_summary, all_api_responses)
    response = {'response_id':_id,'apis':{'data':all_api_responses,'summary':_format_summary(api_summary)},'scores':{'input':scoring_input,'output':scoring_result_json},'offers':{'input':offer_input,'output':offers},'rules':{'input':rule_input,'output':rules_result}, 'decisions':decisions, 'status':pa_status,'code':200,'inputs':input_validation}
    update_response(request_id, _id, _business, _consumers, response, called_api_responses,pa_status,conf['fk'])
    
    return decode_for_json(transform_response_for_user(response, user_access,conf))



state_codes =["AL","AK","AZ","AR","AA","AE","AP","CA","CO","CT","DE","DC","FL","GA",
              "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD","MA","MI","MN","MS",
              "MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA",
              "RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"]
def _check_has_valid_consumer(consumers,validation_msg):
    """ Validates if there is atleast one consumer with SSN and personal details
    Args:
        consumers: list of dict of consumer details
    Returns:
        True/False
    """
    valid_consumers = []
    for consumer in consumers:
        if consumer != None and isinstance(consumer.get('ssn',None),str) :
            valid_consumers.append(consumer)
    if len(valid_consumers) == 0:
         validation_msg = append_validation_msg(validation_msg, 'Valid SSN is required for atleast one consumer')
    
    valid_consumer_msgs = []
    has_valid_consumer = False
    for consumer in valid_consumers:
        consumer_msg = ''
        if set(consumer.keys()) >= set(['name', 'address', 'city', 'state', 'zip', 'ownership_percentage']):
            if is_valid_format(consumer['name'],dict):
                if not is_valid_format(consumer['name']['last'],str):
                    consumer_msg = append_validation_msg(consumer_msg, 'Invalid last name for consumer(s)')
                if not is_valid_format(consumer['name'].get('first',None),str):
                    consumer_msg = append_validation_msg(consumer_msg, 'Invalid first name for consumer(s)')
            else:
                consumer_msg = append_validation_msg(consumer_msg, 'Invalid last/first name for consumer(s)')
            if not is_valid_format(consumer['address'],str):
                consumer_msg = append_validation_msg(consumer_msg, 'Invalid address for consumer(s)')
            if not isNumber(consumer['ownership_percentage']) or float(consumer['ownership_percentage']) <= 0 or float(consumer['ownership_percentage']) > 100:
                consumer_msg = append_validation_msg(consumer_msg, 'Invalid ownership_percentage for consumer(s)')
            if not is_valid_format(consumer['zip'],str) or  len(consumer['zip']) != 5:
                consumer_msg = append_validation_msg(consumer_msg, 'Invalid zip for consumer(s)')
            if not is_valid_format(consumer['city'],str):
                consumer_msg = append_validation_msg(consumer_msg, 'Invalid city for consumer(s)')
            if not is_valid_format(consumer['state'],str) or  consumer['state'].upper() not in state_codes:
                consumer_msg = append_validation_msg(consumer_msg, 'Invalid state for consumer(s). Provide a valid US state code.')
        else:
            consumer_msg = append_validation_msg(consumer_msg, 'Missing mandatory consumer fields: ' + str(set(['name', 'address', 'city', 'state', 'zip', 'ownership_percentage']) - set(consumer.keys())))
        
        if len(consumer_msg) == 0:
            has_valid_consumer = True
            break
        else:
            valid_consumer_msgs.append(consumer_msg)
        
    if not has_valid_consumer:
        for msg in valid_consumer_msgs:
            validation_msg = append_validation_msg(validation_msg, msg)
    return validation_msg

def _check_valid_business(business,validation_msg):
    """ Validates if necessary business details exists
    Args:
        consumers: list of dict of consumer details
    Returns:
        True/False
    """
    if ('dba' not in business or not isinstance(business['dba'],str) or len(business['dba']) == 0) and ('name' not in business or not isinstance(business['name'],str) or len(business['name']) == 0):
        validation_msg = append_validation_msg(validation_msg, 'Business dba or name is required')
    if set(business.keys()) >= set(['address', 'city', 'state', 'zip','business_start_date','email']):
        if not is_valid_format(business['address'],str):
            validation_msg = append_validation_msg(validation_msg, 'Invalid address for business')
        #if not is_valid_format(business.get('phone',None),str) or len(business['phone']) <= 6:
        #    validation_msg = append_validation_msg(validation_msg, 'Invalid phone for business')
        if not is_valid_format(business['business_start_date'],'date'):
            validation_msg = append_validation_msg(validation_msg, 'Invalid business start date')
        if ((not isNumber(business.get('revenue',None))) or (isNumber(business.get('revenue',None)) and float(business.get('revenue')) < 0)) and ((not isNumber(business.get('deposit_amt',None))) or (isNumber(business.get('deposit_amt',None)) and float(business.get('deposit_amt')) < 0)):
            pass
            #validation_msg = append_validation_msg(validation_msg, 'A positive numeric value is required for revenue/deposit amount in business')
        if not is_valid_format(business['zip'],str) or  len(business['zip']) != 5:
            validation_msg = append_validation_msg(validation_msg, 'Invalid zip for business')
        if not is_valid_format(business['city'],str):
            validation_msg = append_validation_msg(validation_msg, 'Invalid city for business')
        if not is_valid_format(business['state'],str) or  business['state'].upper() not in state_codes:
            validation_msg = append_validation_msg(validation_msg, 'Invalid state for business. Provide a valid US state code.')
        if not is_valid_format(business.get('sic',None),str) and not is_valid_format(business.get('naics',None),str):
            if 'industry' in business and is_valid_format(business.get('industry',None),str) and business['industry'] in valid_industry:
                row = industry_naics_mapping[industry_naics_mapping['Industry'] == business['industry']]
                if isinstance(list(row['NAICS'])[0],str):
                    business['naics'] = format_naics(list(row['NAICS'])[0])
                    
                if isinstance(list(row['SIC'])[0],str): 
                    business['sic'] = format_sic(list(row['SIC'])[0])
            else:
                validation_msg = append_validation_msg(validation_msg, 'SIC or NAICS is required for business')
        if not is_valid_format(business.get('email',None),str):
            'Do nothing acceptin input without email'
        else:
            try:
                validate_email(business['email'],check_deliverability=False)
            except EmailSyntaxError:
                validation_msg = append_validation_msg(validation_msg, 'Business email is not in correct format')
            except:
                'Do Nothing for other errors'
    else:
        validation_msg = append_validation_msg(validation_msg, 'Missing mandatory business fields: ' + str(set(['address', 'city', 'state', 'zip','business_start_date','email']) - set(business.keys())))
    return validation_msg


def _transform_request(app_request):
    """ Transforms the request from different structure(application/loan)
    Args:
        app_request: user input in old structure
    Returns:
        user_query - transformed input
    """
    business = {}
    if 'business_overview' in app_request:
        business_overview = app_request['business_overview']
        
        if 'dba' in business_overview:
            business['dba'] = business_overview['dba']
            
        if 'legal_name' in business_overview:
            business['name'] = business_overview['legal_name']
        
         #if 'state_of_incorporation' in business_overview:
         #   business['state'] = business_overview['state_of_incorporation']
        
        if 'federal_id' in business_overview:
            business['fein'] = business_overview['federal_id']
            
        if 'date_established' in business_overview:
            try:
                business['business_start_date'] = reformat_date(business_overview['date_established'], '%Y-%m-%dT%H:%M:%S.%fZ', '%m/%d/%Y')
            except:
                try:
                    business['business_start_date'] = reformat_date(business_overview['date_established'], '%Y-%m-%d', '%m/%d/%Y')
                except:
                    business['business_start_date'] = None
        
        if 'date_current_ownership' in business_overview:
            try:
                business['ownership_start_date'] = reformat_date(business_overview['date_current_ownership'], '%Y-%m-%dT%H:%M:%S.%fZ', '%m/%d/%Y')
            except:
                try:
                    business['ownership_start_date'] = reformat_date(business_overview['date_current_ownership'], '%Y-%m-%d', '%m/%d/%Y')
                except:
                    business['ownership_start_date'] = None
    
        if 'revenue' in business_overview:
            business['revenue'] = business_overview['revenue']
    
        if 'naics' in business_overview:
            business['naics'] = business_overview['naics']
    
    if 'business_location' in app_request:
        business_location = app_request['business_location']
        if 'address' in business_location and len(business_location['address']) > 0:
            address = business_location['address'][0]
            if 'city' in address:
                business['city'] = address['city']
            if 'state' in address:
                business['state'] = address['state']
            if 'postal_code' in address:
                business['zip'] = address['postal_code']
            if 'address' in address:
                business['address'] = address['address']
            
    if 'business_contact' in app_request:
        business_contact = app_request['business_contact']
        if 'phone' in business_contact:
            business['phone'] = business_contact['phone']
        if 'email' in business_contact:
            business['email'] = business_contact['email']
        if 'website' in business_contact:
            business['website'] = business_contact['website']
        
    consumers = []
    if 'principals' in app_request:
        if isinstance(app_request['principals'],list):
            for principal in app_request['principals']:
                consumer = {}
                if 'percent_ownership' in principal:
                    consumer['ownership_percentage'] = principal['percent_ownership']
                
                if 'dob' in principal:
                    try:
                        consumer['dob'] = reformat_date(principal['dob'], '%Y-%m-%dT%H:%M:%S.%fZ', '%m/%d/%Y')
                    except:
                        try:
                            consumer['dob'] = reformat_date(principal['dob'], '%Y-%m-%d', '%m/%d/%Y')
                        except:
                            consumer['dob'] = None
                
                if 'ssn' in principal:
                    consumer['ssn'] = principal['ssn']
                
                if 'permissible_purpose' in principal:
                    consumer['credit_pull_consent'] = principal['permissible_purpose']
                
                if 'address' in principal:
                    _address = principal['address']
                    if 'address' in _address:
                        consumer['address'] = _address['address']
                    if 'city' in _address:
                        consumer['city'] = _address['city']
                    if 'state' in _address:
                        consumer['state'] = _address['state']
                    if 'postal_code' in _address:
                        consumer['zip'] = _address['postal_code']
                        
                if 'name_last' in principal:
                    consumer['name'] = {
                                        'last':principal['name_last'],
                                        'first':principal.get('name_first',None),
                                        'middle':principal.get('name_middle',None),
                                        'suffix':principal.get('suffix',None),
                                        'prefix':principal.get('prefix',None),
                                        }
                    consumer['last_name'] = principal['name_last']
                    consumer['first_name'] = principal.get('name_first',None)
                    consumer['middle_name'] = principal.get('name_middle',None)
                    consumer['suffix_name'] = principal.get('suffix',None)
                    consumer['prefix_name'] = principal.get('prefix',None)
                consumers.append(consumer)
    
        
    _query = {'business':business, 'consumers':consumers, 'submitted_application':app_request}
    return _query

def append_validation_msg(validation_msg, msg):
    if len(validation_msg) > 0:
        validation_msg += ', ' + msg
    else:
        validation_msg = msg
    return validation_msg

def check_valid_request(_input):
    user_query = {}
    if _input is None:
        _input = {}
    
    if isinstance(_input,dict) and 'business_overview' in _input and 'principals' in _input:
        user_query = _transform_request(_input)
    else:
        user_query = _input
        
    
    validation_msg = ''
    if user_query == None or not isinstance(user_query,dict):
        validation_msg = 'Input need to be json'
    if not isinstance(user_query.get('business',None),dict):
        validation_msg = append_validation_msg(validation_msg, 'business information is missing')
    if not isinstance(user_query.get('consumers',None), list) or len(user_query.get('consumers',[])) == 0:
        validation_msg = append_validation_msg(validation_msg, 'consumers information is missing')
    
    
    consumers = user_query.get('consumers',[])
    if isinstance(consumers, dict):
        consumers = [consumers]
    
    if len(validation_msg) == 0:
        user_query = standardize_input(user_query)
        
        validation_msg = _check_has_valid_consumer(user_query.get('consumers',[]), validation_msg)
        validation_msg = _check_valid_business(user_query.get('business',{}), validation_msg)
        if len(validation_msg) == 0:
            return True
        else:
            return False
    else:
        return False

def _remove_null_and_empty(_input):
    if isinstance(_input,dict):
        items_to_pop = []
        for item in _input:
            if _input[item] is None:
                items_to_pop.append(item)
            elif isinstance(_input[item],str) and len(_input[item].strip()) == 0:
                items_to_pop.append(item)
            elif isinstance(_input[item],list):
                _input[item] = _remove_null_and_empty(_input[item])
            elif isinstance(_input[item],dict):
                _input[item] = _remove_null_and_empty(_input[item])
        if len(items_to_pop) > 0:
            for item in items_to_pop:
                _input.pop(item)
                
    if isinstance(_input,list):
        items_to_remove = []
        for ix in range(len(_input)):
            item = _input[ix]
            if item is None:
                items_to_remove.append(item)
            elif isinstance(item,str) and len(item.strip()) == 0:
                items_to_remove.append(item)
            elif isinstance(item,list):
                _input[ix] = _remove_null_and_empty(item)
            elif isinstance(item,dict):
                _input[ix] = _remove_null_and_empty(item)
        
        if len(items_to_remove) > 0:
            for item in items_to_remove:
                _input.remove(item)
        
    return _input                    
                
            
    
def validate_input_fields(user_query):
    """
    Provides a summary of missing and invalid format of input fields for optional and madatory fields
    """
    user_query = _remove_null_and_empty(user_query)
    if user_query == None:
        user_query = {}
    if not isinstance(user_query.get('business', None), dict):
        user_query['business'] = {}
    if not isinstance(user_query.get('consumers', None), list):
        user_query['consumers'] = [{}]
    missing_business = list(set(['dba','name','fein','address', 'city', 'state','phone', 'zip','business_start_date','email','website','naics','sic','revenue','deposit_amt','ownership_start_date']) - set(user_query.get('business').keys()))
    invalid_business = []
    business = user_query.get('business')
    if 'dba' in business and not is_valid_format(business.get('dba',None),str):
        invalid_business.append('dba')
    if 'name' in business and not is_valid_format(business.get('name',None),str):
        invalid_business.append('name')
    if 'fein' in business and (not is_valid_format(business.get('fein',None),str) or not validate_fein(business['fein'])):
        invalid_business.append('fein')
    if 'address' in business and not is_valid_format(business.get('address',None),str):
        invalid_business.append('address')
    if 'phone' in business and (not is_valid_format(business.get('phone',None),str) or len(business['phone']) <= 6):
        invalid_business.append('phone')
    if 'ownership_start_date' in business and not is_valid_format(business.get('ownership_start_date',None),'date'):
        invalid_business.append('ownership_start_date')
    if 'business_start_date' in business and not is_valid_format(business.get('business_start_date',None),'date'):
        invalid_business.append('business_start_date')
    if 'revenue' in business and (not isNumber(business.get('revenue',None)) or float(business.get('revenue')) <= 0): 
        invalid_business.append('revenue')
    if 'deposit_amt' in business and (not isNumber(business.get('deposit_amt',None)) or float(business.get('deposit_amt')) <= 0):
        invalid_business.append('deposit_amt')
    if 'zip' in business and (not is_valid_format(business.get('zip',None),str) or  len(format_zip(business['zip'])) != 5):
        invalid_business.append('zip')
    if 'city' in business and not is_valid_format(business.get('city',None),str):
        invalid_business.append('city')
    if 'state' in business and (not is_valid_format(business.get('state',None),str) or  business['state'].upper() not in state_codes):
        invalid_business.append('state')
    if 'website' in business and (not is_valid_format(business.get('website',None),str) or  '.' not in business['website']):
        invalid_business.append('website')
    if 'email' in business and not is_valid_format(business.get('email',None),str):
        invalid_business.append('email')
    elif 'email' in business:
        try:
            validate_email(business['email'],check_deliverability=False)
        except:
            invalid_business.append('email')
    if 'sic' in business and (not is_valid_format(business.get('sic',None),str) or not isNumber(format_sic(business['sic']))):
        invalid_business.append('sic')
    if 'naics' in business and (not is_valid_format(business.get('naics',None),str) or not isNumber(format_naics(business['naics']))):
        invalid_business.append('naics')
    if 'industry' in business and (not is_valid_format(business.get('industry',None),str) or business['industry'] not in valid_industry):
        invalid_business.append('industry')
    invalid_consumers = []
    missing_consumers = []
    for consumer in user_query['consumers']:
        if not isinstance(consumer,dict):
            consumer = {}
        missing_fields = list(set(['ssn','first_name', 'last_name','middle_name','address', 'city', 'state', 'zip', 'ownership_percentage','email','phone','dob']) - set(consumer.keys()))
        if len(missing_fields) > 0:
            missing_consumers.append(missing_fields)
        
        invalid_fields = []
        
        if 'ssn' in consumer and not is_valid_ssn(consumer.get('ssn',None)):
            invalid_fields.append('ssn')
        if 'first_name' in consumer and not is_valid_format(consumer.get('first_name',None),str):
            invalid_fields.append('first_name')
        if 'last_name' in consumer and not is_valid_format(consumer.get('last_name',None),str):
            invalid_fields.append('last_name')
        if 'middle_name' in consumer and not is_valid_format(consumer.get('middle_name',None),str):
            invalid_fields.append('middle_name')
        if 'ownership_percentage' in consumer and (not isNumber(consumer.get('ownership_percentage',None)) or float(consumer['ownership_percentage']) <= 0 or float(consumer['ownership_percentage']) > 100):
            invalid_fields.append('ownership_percentage')
        if 'address' in consumer and not is_valid_format(consumer.get('address',None),str):
            invalid_fields.append('address')
        if 'phone' in consumer and (not is_valid_format(consumer.get('phone',None),str) or len(consumer['phone']) <= 6):
            invalid_fields.append('phone')
        if 'dob' in consumer and not is_valid_format(consumer.get('dob',None),'date'):
            invalid_fields.append('dob')
        if 'zip' in consumer and (not is_valid_format(consumer.get('zip',None),str) or  len(format_zip(consumer['zip'])) != 5):
            invalid_fields.append('zip')
        if 'city' in consumer and not is_valid_format(consumer.get('city',None),str):
            invalid_fields.append('city')
        if 'state' in consumer and (not is_valid_format(consumer.get('state',None),str) or  consumer['state'].upper() not in state_codes):
            invalid_fields.append('state')
        if 'signer_for_contracts' in consumer and not is_valid_format(consumer.get('signer_for_contracts',None),bool):
            invalid_fields.append('signer_for_contracts')
        if 'bank_info_responsible' in consumer and not is_valid_format(consumer.get('bank_info_responsible',None),bool):
            invalid_fields.append('bank_info_responsible')
        if 'credit_pull_consent' in consumer and not is_valid_format(consumer.get('credit_pull_consent',None),bool):
            invalid_fields.append('credit_pull_consent')
        
        if 'email' in consumer and not is_valid_format(consumer.get('email',None),str):
            email_valid = False
            try:
                validate_email(consumer['email'],check_deliverability=False)
            except:
                invalid_fields.append('email')    
            
        if len(invalid_fields) > 0:
            invalid_consumers.append(invalid_fields)
            
    input_validation = None
    if len(missing_business) > 0 or len(missing_consumers) > 0 or len(invalid_business) > 0 or len(invalid_consumers) > 0:
        input_validation = {}
        if len(missing_business) > 0 or len(missing_consumers) > 0:
            input_validation['missing'] = {'business':missing_business,'consumers':missing_consumers}
        
        if len(invalid_business) > 0 or len(invalid_consumers) > 0:
            input_validation['invalid']={'business':invalid_business,'consumers':invalid_consumers}
    return input_validation
    
    
    
    
# This function is a wrapper for preapproval - palceholder to refactor if the host for the service changes
def preapproval_handler(_input, channel, cached_objects, uid=None,user_id=None,update=False):
    """ Handler function for preapproval process
    Args:
        user_query: dict/json of input. Flask gives dict
        _id: unique preapproval identifier. Use this to rerun or update existing requests.
    Returns:
        result of run_preapproval or error codes
    """
    user_query = {}
    if _input is None:
        _input = {}
    
    if isinstance(_input,dict) and 'business_overview' in _input and 'principals' in _input:
        user_query = _transform_request(_input)
    else:
        user_query = _input
    input_validation = validate_input_fields(user_query)
    validation_msg = ''
    if user_query == None or not isinstance(user_query,dict):
        validation_msg = 'Input need to be json'
    if not isinstance(user_query.get('business',None),dict):
        validation_msg = append_validation_msg(validation_msg, 'business information is missing')
    if not isinstance(user_query.get('consumers',None), list) or len(user_query.get('consumers',[])) == 0:
        validation_msg = append_validation_msg(validation_msg, 'consumers information is missing')
    
    consumers = user_query.get('consumers',[])
    if isinstance(consumers, dict):
        consumers = [consumers]
    
    if len(validation_msg) == 0:
        user_query = standardize_input(user_query)
        validation_msg = _check_has_valid_consumer(user_query.get('consumers',[]), validation_msg)
        validation_msg = _check_valid_business(user_query.get('business',{}), validation_msg)
        if len(validation_msg) == 0:
            user_query['response_id'] = uid
            _id,conf,user_access = create_request_id(uid, user_id, user_query,channel, update)
            user_query['request_id'] = _id
            if not isinstance(user_query.get('apis',None),list) or len(user_query['apis']) == 0:
                user_query['apis'] = conf['apis']
            try:
                is_dupe=dupe_app_search(user_query)
                print(str(is_dupe))
            except Exception as e:
                is_dupe={}
                print(str(e))
                
            if is_dupe.get('Dupe',False)==True:
                if is_dupe.get("Dupe_type","")=="RC":
                    response = {'status':1,'code':200,'response_id':uid,'decisions':{'decision': 'Not prequalified','message':is_dupe.get('message',validation_msg)+" ,ReCap ID : "+str(set(is_dupe.get('Dupe_id',[]))),'decline_reasons':'Duplicate Application'}}
                else:
                    response = {'status':1,'code':200,'response_id':uid,'decisions':{'decision': 'Not prequalified','message':is_dupe.get('message',validation_msg)+" ,PQ ID : "+str(set(is_dupe.get('Dupe_id',[]))),'decline_reasons':'Duplicate Application'}}
                resp_to_persist = dict(response)
                resp_to_persist['inputs'] = input_validation
                update_response(_id,uid, None, None, response, {},1,conf['fk'])
                return response
            else:    
                return run_preapproval({'business':user_query['business'], 'consumers':user_query['consumers']},user_query['apis'],_id, uid,conf,user_id,input_validation,user_access,cached_objects,update)
    
    user_query['response_id'] = uid
    _id,conf,user_access = create_request_id(uid, user_id, user_query,channel, update)
    response = {'status':99,'code':400,'message':validation_msg,'response_id':uid}
    resp_to_persist = dict(response)
    resp_to_persist['inputs'] = input_validation
    update_response(_id,uid, None, None, response, {},99,conf['fk'])
    return response
    
    
