#This is application to find roots of a cubic equation using Vieta's substitution

#importing external packages
import math
import json

#importing parts of code written in the outside files
import logger
import parser

logger.logging.info("Start program...")

#parsing path to the file with json, deserialising and computing params
args = parser.parse_args()

try:
    path = args.path
    with open(path,"r") as json_file:
        params = json.load(json_file)
except:
    logger.logging.error("No file with params!")
else:
    try:
        a, b, c = params["b"]/params["a"], params["c"]/params["a"], params["d"]/params["a"]
        logger.logging.info("Params loaded!")
    except:
        logger.logging.error("Invalid params in the file!")
    else:


        try:
            # calculating substitute values
            q = (pow(a,2)-3*b)/9
            r = (2*pow(a,3)-9*a*b+27*c)/54
            s = pow(q,3)-pow(r,2)
            logger.logging.info(f"q = {q}")
            logger.logging.info(f"r = {r}")
            logger.logging.info(f"s = {s}")
            #processing three possible ranges of s
            if s > 0:
                f = 1/3 * math.acos(r/math.sqrt(pow(q,3)))
                x1 = -2*math.sqrt(q)*math.cos(f) - a/3
                x2 = -2*math.sqrt(q)*math.cos(f+2/3*math.pi) - a/3
                x3 = -2*math.sqrt(q)*math.cos(f-2/3*math.pi) - a/3
                logger.logging.info("Three real roots:")
                logger.logging.info(f"x1 = {x1}")
                logger.logging.info(f"x2 = {x2}")
                logger.logging.info(f"x3 = {x3}")

            if s < 0:
                # for s below zero processing three possible ranges of q
                if q > 0:
                    f = 1 / 3 * math.acosh(abs(r)/math.sqrt(pow(q, 3)))
                    x1 = -2 *math.copysign(1,r)*math.sqrt(q) * math.cosh(f)-a/3
                    x2 = math.copysign(1,r)*math.sqrt(q) * math.cosh(f)-a/3+1j*math.sqrt(3)*math.sqrt(q)*math.sinh(f)
                    x3 = math.copysign(1,r)*math.sqrt(q) * math.cosh(f)-a/3-1j*math.sqrt(3)*math.sqrt(q)*math.sinh(f)
                    logger.logging.info("One real root and two complex ones:")
                    logger.logging.info(f"x1 = {x1}")
                    logger.logging.info(f"x2 = {x2}")
                    logger.logging.info(f"x3 = {x3}")

                if q < 0:
                    q=abs(q)
                    f = 1 / 3 * math.asinh(abs(r)/math.sqrt(pow(q, 3)))
                    x1 = -2 *math.copysign(1,r)*math.sqrt(q) * math.sinh(f)-a/3
                    x2 = math.copysign(1,r)*math.sqrt(q) * math.sinh(f)-a/3+1j*math.sqrt(3)*math.sqrt(q)*math.cosh(f)
                    x3 = math.copysign(1,r)*math.sqrt(q) * math.sinh(f)-a/3-1j*math.sqrt(3)*math.sqrt(q)*math.cosh(f)
                    logger.logging.info("One real root and two complex ones:")
                    logger.logging.info(f"x1 = {x1}")
                    logger.logging.info(f"x2 = {x2}")
                    logger.logging.info(f"x3 = {x3}")

                if q == 0:
                    x1 = -1*(c-pow(a,3)/27)**(1./3.)-a/3
                    x2 = -1*(a+x1)/2+1j/2*math.sqrt(abs((a-3*x1)*(a+x1)-4*b))
                    x2 = -1*(a+x1)/2-1j/2*math.sqrt(abs((a-3*x1)*(a+x1)-4*b))
                    logger.logging.info("One real root and two complex ones:")
                    logger.logging.info(f"x1 = {x1}")
                    logger.logging.info(f"x2 = {x2}")
                    logger.logging.info(f"x3 = {x3}")

            if s == 0:
                x1 = -2*r**(1./3.)-a/3
                x2 = r**(1./3.)-a/3
                logger.logging.info("Two real roots:")
                logger.logging.info(f"x1 = {x1}")
                logger.logging.info(f"x2 = {x2}")

        except Exception as e:
            logger.logging.error("Exception during calculation occurred!",exc_info=True)
