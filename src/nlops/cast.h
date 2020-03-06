
struct linop_s;
extern struct nlop_s* nlop_from_linop(const struct linop_s*);
extern struct nlop_s* nlop_from_linop_F(const struct linop_s*);
extern const struct linop_s* linop_from_nlop(const struct nlop_s* x);
