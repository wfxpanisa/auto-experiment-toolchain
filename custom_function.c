#define _GNU_SOURCE
#include "gtk_logger.h"
#include <stdarg.h>
#include <dlfcn.h>

int c = 0;
void gtk_widget_show (GtkWidget *widget)
{
        static char * (*func)();
        if(!func)
                func = (char *(*)()) dlsym(RTLD_NEXT, "gtk_widget_show");
        printf("Overridden: %d  - %s\n", c++, gtk_widget_get_name(widget) );
        func(widget);
        return;
}