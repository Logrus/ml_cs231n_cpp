#include <classifiers/simplenet_ui.h>
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    SimpleNetUI w;
    w.show();

    return a.exec();
}
