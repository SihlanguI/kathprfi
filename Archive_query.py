import katdal
import numpy as np
import pandas as pd
import pysolr
import argparse

class SARAOArchiveQuery:
    def __init__(self, start_time, end_time, band, dump_rate, channel_mode):
        self.start_time = start_time
        self.end_time = end_time
        self.band = band
        self.dump_rate = dump_rate
        self.channel_mode = channel_mode
        self.archive = pysolr.Solr('http://kat-archive.kat.ac.za:8983/solr/kat_core')

    @staticmethod
    def arg_parser():
        parser = argparse.ArgumentParser(description='SARAO Archive Query')
        parser.add_argument('-s', '--start_time', type=str, required=True, help='Start time (format: YYYY-MM-DDTHH:MM:SSZ)')
        parser.add_argument('-e', '--end_time', type=str, required=True, help='End time (format: YYYY-MM-DDTHH:MM:SSZ)')
        parser.add_argument('--band', type=str, choices=['L', 'U'], default='U', help='Band of interest (L or U)')
        parser.add_argument('-t', '--dump_rate', type=float, default=7.9966, help='Dump rate')
        parser.add_argument('-f', '--channel_mode', type=int, default=4096, help='Number of frequency channels')
        return parser
    
    def search_archive(self):
        search_str = ['CAS.ProductTypeName:MeerKATTelescopeProduct',
                      f'StartTime:[{self.start_time} TO {self.end_time}]']
        search_string = ' AND '.join(search_str)
        ar_res = self.archive.search(search_string, rows=10000, **{'sort': 'StartTime desc'})
        search_results = ar_res.docs
        return search_results

    def process_results(self, search_results):
        imaging_links = []
        imaging_info = []

        #Define band-specific parametrs based on the parsed arguments
        if self.band == "U":
            target_center_freq = 816132812
            target_bandwidth = 544e6
        elif self.band == "L":
            target_center_freq = 1284e6
            target_bandwidth = 856e6

        for observation in search_results[::-1]:
            if 'CaptureBlockId' in observation and observation['ProposalId'][:3] == 'SCI':
                center_freq = round(observation['CenterFrequency'] + observation['ChannelWidth'])
                bandwidth = observation['Bandwidth']

                if center_freq == target_center_freq and bandwidth == target_bandwidth:
                    filename = f"http://archive-gw-1.kat.ac.za:7480/{observation['CaptureBlockId']}/{observation['CaptureBlockId']}_sdp_l0.full.rdb"
                    imaging_links.append(filename)
                    imaging_info.append(katdal.open(filename))
                    print(filename)
        return imaging_links, imaging_info

    def save_to_csv(self, imaging_links, imaging_info):
        imagingdf = pd.DataFrame({'FullLink': imaging_links})
        imagingdf.to_csv(f"sci_Imaging_{self.band}_{self.start_time}_{self.end_time}.csv", index=False)
        imaging_info_df = pd.DataFrame(imaging_info)
        imaging_info_df.to_csv(f"sci_Imaging_full_{self.band}_{self.start_time}_{self.end_time}.csv", index=False)

if __name__ == "__main__":
    #create parser arguments
    parser = SARAOArchiveQuery.arg_parser()
    args = parser.parse_args()

    # Create SARAOArchiveQuery instance and perform query and processing
    archive_query = SARAOArchiveQuery(args.start_time, args.end_time, args.band, args.dump_rate, args.channel_mode)
    search_results = archive_query.search_archive()
    imaging_links, imaging_info = archive_query.process_results(search_results)
    archive_query.save_to_csv(imaging_links, imaging_info)

'''
When executing the script, you can do;
ipython Archive_query.py -- --start_time '2023-12-01T00:00:00Z' --end_time '2023-12-31T00:00:00Z' --band "L" -t 8 -f 4096
'''


