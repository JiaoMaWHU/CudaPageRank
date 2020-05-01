# Copyright (c) 2010, Panos Louridas, GRNET S.A.
#
#  All rights reserved.
# 
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#
#  * Redistributions of source code must retain the above copyright
#  notice, this list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright
#  notice, this list of conditions and the following disclaimer in the
#  documentation and/or other materials provided with the
#  distribution.
#
#  * Neither the name of GRNET S.A, nor the names of its contributors
#  may be used to endorse or promote products derived from this
#  software without specific prior written permission.
# 
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
#  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#  COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
#  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
#  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
#  STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
#  OF THE POSSIBILITY OF SUCH DAMAGE.

import sys, time
from pageRank import pageRank

links = [[]]


def read_file(filename):
    f = open(filename, 'r')
    for line in f:
        (frm, to) = map(int, line.split(" "))
        extend = max(frm - len(links), to - len(links)) + 1
        for i in range(extend):
            links.append([])
        links[frm].append(to)
    f.close()


fn = "1000.txt"
read_file(fn)

f = open("time-%s" % fn, 'w')
for i in range(5):
    start = int(round(time.time() * 1000))
    pr = pageRank(links, alpha=0.85, convergence=0.00001, checkSteps=10)
    used = int(round(time.time() * 1000)) - start
    f.writelines(["no.%d time used: %s ms\n" %(i,used)])
f.close()

# sum = 0
# for i in range(len(pr)):
#     print i, "=", pr[i]
#     sum = sum + pr[i]
# print "s = " + str(sum)
