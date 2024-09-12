#include <unistd.h>
#include <X11/Xlib.h>

extern Display *display;
extern Window window;
extern GC gc;
extern int screen;

void create_display(int x, int y, int height, int width);
void draw_point(int x, int y, int color);
void flush();