from typing import Union, List, Tuple, Optional, Any
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
    expr = expr.lower()
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
        elif c == '&':
            if current is not None and current != '&':
                result.append(current)
                current = None
            
            if current is None:
                current = ''

            current += '&'
            
            if current == '&&':
                result.append('&&')
                current = None
        elif c == '|':
            if current is not None and current != '|':
                result.append(current)
                current = None
            
            if current is None:
                current = ''

            current += '|'
            
            if current == '||':
                result.append('||')
                current = None
        elif c == '=':
            if current is not None and current != '=':
                result.append(current)
                current = None
            
            if current is None:
                current = ''

            current += '='
            
            if current == '==':
                result.append('==')
                current = None
        elif c in ('+', '-', '*', '/', ',', '?', ':', '<', '>'):
            if current is not None:
                result.append(current)
                current = None

            result.append(c)
        elif c.isalnum() or c == '_':
            if current is None:
                current = ''
            if '=' in current or '&' in current or '|' in current:
                result.append(current)
                current = ''

            current += c
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
                rhs = []

                while True:
                    rhs.append(parse(tokens, 0))
                    nextTok = tokens.next()
                    if nextTok == ',':
                        pass
                    elif nextTok == ')':
                        break
                    else:
                        raise Exception(nextTok)
                lhs = (lhs, tuple(rhs))
            else:
                lhs = (op, (lhs, ))

            continue
    
        ifp = infix_power(op)
        if ifp is not None:
            (leftIfp, rightIfp) = ifp
            if leftIfp < min_pow:
                break
        
            tokens.next()

            if op == '?':
                mhs = parse(tokens, 0)
                assert(tokens.next() == ':')
                rhs = parse(tokens, rightIfp)
                lhs = (op, (lhs, mhs, rhs))
            else:
                rhs = parse(tokens, rightIfp)
                lhs = (op, (lhs, rhs))

            continue
    
        break
    
    return lhs

def prefix_power(c: Token) -> Optional[int]:
    if c == 'return':
        return 1
    if c in ('+', '-'):
        return 17
    else:
        return None

def postfix_power(c: Token) -> Optional[int]:
    if c == '(':
        return 15

def infix_power(c: Token) -> Optional[Tuple[int, int]]:
    if c == '?':
        return (4, 3)
    elif c == '||':
        return (5, 6)
    elif c == '&&':
        return (7, 8)
    elif c in ('<', '>', '=='):
        return (9, 10)
    elif c in ('+', '-'):
        return (11, 12)
    elif c in ('*', '/'):
        return (13, 14)
    else:
        return None

def printTree(p, indent=0):
    if isinstance(p, tuple):
        for elem in p:
            printTree(elem, indent+2)
    else:
        print(f"{' '*indent}{p}")

def parseStr(s: str) -> List[Any]:
    if ';' in s:
        result = []
        for line in s.split(';')[:-1]:
            result.append(parse(TokenIter(lex(line), 0)))
        return result
    else:
        return [parse(TokenIter(lex(s), 0))]

def evalString(s, entity):
    try:
        assert(';' not in s)
        parsed = parseStr(s)
        result = evalOneExpr(parsed[0], entity)
        assert(result is not None)
        return result
    except Exception as e:
        print(s)
        print(lex(s))
        raise Exception((e, s))
    
def evalExpr(p, entity):
    for expr in p[:-1]:
        evalOneExpr(expr, entity)
    
    return evalOneExpr(p[-1], entity)

def evalOneExpr(p, entity):
    def evalApplied(p):
        return evalOneExpr(p, entity)

    if isinstance(p, tuple):
        (op, args) = p

        if op == '=':
            assert(args[0].startswith('variable'))
            entity.variables[args[0].removeprefix('variable')] = evalApplied(args[1])
            return None

        args = tuple(map(evalApplied, args))

        OPS = [
            # Unary
            {
                '+': lambda x: x,
                '-': lambda x: -x,
                'math.sin': lambda x: math.sin(math.radians(x)),
                'math.cos': lambda x: math.cos(math.radians(x)),
                'math.sqrt': math.sqrt,
                'math.round': round,
                'query.position_delta': lambda x: entity.getQuery(f'position_delta{int(x)}'),
                'return': lambda x: x,
            },
            # Binary
            {
                '+': operator.add,
                '-': operator.sub,
                '*': operator.mul,
                '/': operator.truediv,
                '<': operator.lt,
                '>': operator.gt,
                '&&': lambda a, b: 1.0 if a != 0 and b != 0 else 0.0,
                '||': lambda a, b: 1.0 if a != 0 or b != 0 else 0.0,
                '==': lambda a, b: 1.0 if a == b else 0.0,
                'math.pow': math.pow,
                'math.mod': lambda x, y: x % y,
            },
            # Ternary
            {
                'math.lerp': lambda a, b, t: a * (1 - t) + b * t,
                'math.clamp': lambda x, l, h: max(min(x, h), l),
                '?': lambda c, t, f: t if c != 0 else f,
            },
        ]

        try:
            ops = OPS[len(args) - 1]
        except IndexError:
            raise Exception(args)

        return ops[op](*args)
    elif isinstance(p, float) or isinstance(p, int):
        return float(p)
    elif p.startswith('query.'):
        p = p.removeprefix('query.')
        return entity.getQuery(p)
    elif p.startswith('variable.'):
        p = p.removeprefix('variable.')
        if p not in entity.variables:
            print(f"UNKNOWN VARIABLE {p}")
            if p == 'gliding_speed_value':
                entity.variables[p] = 1.0
            else:
                entity.variables[p] = 0.0
        
        return entity.variables[p]
    elif p == 'this':
        # TODO:
        return 0.0
    else:
        raise Exception(repr(p))