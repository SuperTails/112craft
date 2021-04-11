from typing import Union, List, Tuple, Optional
from dataclasses import dataclass
import operator
import math

# a literal, a parenthesis, 
Token = Union[float, str]

def couldBeFloat(s: str) -> bool:
    seenDecimal = False

    for c in s:
        if c == '.':
            if seenDecimal:
                return False
            else:
                seenDecimal = True
        elif not c.isdigit():
            return False
    
    return True

def lex(expr: str) -> List[Token]:
    result = []
    current = None
    for c in expr:
        if c in ('(', ')'):
            if current is not None:
                result.append(current)
                current = None

            result.append(c)
        elif c.isspace():
            if current is not None:
                result.append(current)
                current = None
        elif c == '.':
            if current is None:
                current = ''
            current += c
        elif c.isalnum() or c == '_':
            if current is None:
                current = ''

            current += c
        elif c in ('+', '-', '*', '/'):
            if current is not None:
                result.append(current)
                current = None

            result.append(c)
        else:
            raise Exception(c)
    
    if current is not None:
        result.append(current)
        current = None
    
    for i in range(len(result)):
        if isinstance(result[i], str) and couldBeFloat(result[i]):
            result[i] = float(result[i])
    
    return result

# https://matklad.github.io/2020/04/13/simple-but-powerful-pratt-parsing.html

@dataclass
class TokenIter:
    tokens: List[Token]
    i: int

    def next(self) -> Optional[Token]:
        if self.count() == 0:
            return None
        else:
            result = self.tokens[self.i]
            self.i += 1
            return result
    
    def peek(self) -> Optional[Token]:
        if self.count() == 0:
            return None
        else:
            return self.tokens[self.i]
    
    def count(self) -> int:
        return len(self.tokens) - self.i
    
def parse(tokens: TokenIter, min_pow=0):
    lhs = tokens.next()
    if lhs is None:
        raise Exception("Unexpected end of input")
    elif lhs == '(':
        lhs = parse(tokens, 0)
        assert(tokens.next() == ')')
    elif lhs == ')':
        3 / 0
    elif lhs in ('+', '-'):
        pfp = prefix_power(lhs)
        rhs = parse(tokens, pfp)
        lhs = (lhs, (rhs, ))
    
    while True:
        op = tokens.peek()
        if op is None:
            break
        
        pfp = postfix_power(op)
        if pfp is not None:
            if pfp < min_pow:
                break

            tokens.next()

            if op == '(':
                rhs = parse(tokens, 0)
                assert(tokens.next() == ')')
                lhs = (lhs, (rhs, ))
            else:
                lhs = (op, (lhs, ))

            continue
    
        ifp = infix_power(op)
        if ifp is not None:
            (leftIfp, rightIfp) = ifp
            if leftIfp < min_pow:
                break
        
            tokens.next()

            rhs = parse(tokens, rightIfp)
            lhs = (op, (lhs, rhs))

            continue
    
        break
    
    return lhs

def prefix_power(c: Token) -> Optional[int]:
    if c in ('+', '-'):
        return 9
    else:
        return None

def postfix_power(c: Token) -> Optional[int]:
    if c == '(':
        return 7

def infix_power(c: Token) -> Optional[Tuple[int, int]]:
    if c in ('+', '-'):
        return (1, 2)
    elif c in ('*', '/'):
        return (3, 4)
    else:
        return None

def printTree(p, indent=0):
    if isinstance(p, tuple):
        for elem in p:
            printTree(elem, indent+2)
    else:
        print(f"{' '*indent}{p}")

def evalString(s, entity):
    parsed = parse(TokenIter(lex(s), 0))
    return evalExpr(parsed, entity)

def evalExpr(p, entity):
    def evalApplied(p):
        return evalExpr(p, entity)

    if isinstance(p, tuple):
        (op, args) = p

        args = tuple(map(evalApplied, args))

        UNARY_OPS = {
            '+': lambda x: x,
            '-': lambda x: -x,
            'math.sin': lambda x: math.sin(math.radians(x)),
            'math.cos': lambda x: math.cos(math.radians(x)),
        }

        BINARY_OPS = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
        }

        if len(args) == 1:
            ops = UNARY_OPS
        elif len(args) == 2:
            ops = BINARY_OPS
        else:
            raise Exception(args)

        return ops[op](*args)
    elif isinstance(p, float):
        return p
    elif p.startswith('query.'):
        p = p.removeprefix('query.')
        return entity.getQuery(p)
    elif p.startswith('variable.'):
        p = p.removeprefix('variable.')
        if p not in entity.variables:
            print("UNKNOWN VARIABLE {p}")
            entity.variables[p] = 0.0
        
        return entity.variables[p]
    elif p == 'this':
        # TODO:
        return 0.0
    else:
        raise Exception(p)